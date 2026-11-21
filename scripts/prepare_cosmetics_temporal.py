"""Build a temporal split for a fixed cohort of active REES46 Cosmetics users."""

import argparse
import json
from pathlib import Path

import duckdb


def prepare(raw: Path, output: Path, database: Path, max_users: int) -> None:
    if max_users < 1:
        raise ValueError("max_users must be positive")
    if output.exists() or database.exists():
        raise FileExistsError("Choose new output and database paths")
    files = [str(path) for path in sorted(raw.glob("*.csv"))]
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw}")
    output.mkdir(parents=True)
    with duckdb.connect(str(database)) as con:
        con.execute("SET memory_limit='3GB'")
        con.execute("SET threads=4")
        con.execute("SET preserve_insertion_order=false")
        con.execute("""
            CREATE TABLE events AS
            SELECT event_time, event_type, product_id::BIGINT AS item_id,
                   category_id AS category, user_id, trim(user_session) AS session_id
            FROM (
                SELECT DISTINCT * FROM read_csv(?, header=true, all_varchar=true)
                WHERE event_time IS NOT NULL AND event_type IS NOT NULL
                  AND product_id IS NOT NULL AND user_id IS NOT NULL
                  AND nullif(trim(user_session), '') IS NOT NULL
            )
        """, [files])
        n_events = con.execute("SELECT count(*) FROM events").fetchone()[0]
        cuts = con.execute("""
            SELECT list(event_time ORDER BY event_time)
            FROM (
                SELECT event_time, row_number() OVER (ORDER BY event_time) AS pos FROM events
            ) WHERE pos IN (?, ?)
        """, [int(n_events * 0.8) + 1, int(n_events * 0.9) + 1]).fetchone()[0]
        if len(cuts) != 2 or cuts[0] >= cuts[1]:
            raise ValueError("Temporal cutoffs must be distinct")
        print(f"Events with a session: {n_events:,}; cutoffs: {cuts}", flush=True)
        con.execute("""
            CREATE TABLE sessions AS
            WITH boundaries AS (
                SELECT user_id, session_id, min(event_time) AS first, max(event_time) AS last,
                       count(DISTINCT item_id) FILTER (WHERE event_type = 'purchase') AS buy_items
                FROM events GROUP BY user_id, session_id
            )
            SELECT user_id, session_id, buy_items,
                   CASE WHEN last < ? THEN 'train'
                        WHEN first >= ? THEN 'test'
                        WHEN first >= ? AND last < ? THEN 'validation' END AS phase
            FROM boundaries
        """, [cuts[0], cuts[1], cuts[0], cuts[1]])
        con.execute("""
            CREATE TABLE cohort AS
            SELECT user_id, count(*) AS sessions
            FROM sessions
            WHERE phase = 'train' AND buy_items BETWEEN 2 AND 30
            GROUP BY user_id ORDER BY sessions DESC, user_id LIMIT ?
        """, [max_users])
        con.execute("""
            CREATE TABLE session_items AS
            WITH dedup AS (
                SELECT e.user_id, e.session_id, s.phase, e.event_type, e.item_id,
                       first(category ORDER BY event_time DESC, category ASC) AS category
                FROM events e JOIN cohort c USING (user_id)
                JOIN sessions s USING (user_id, session_id)
                WHERE s.phase IS NOT NULL AND e.event_type IN ('purchase', 'view')
                GROUP BY ALL
            )
            SELECT *, count(*) OVER (PARTITION BY user_id, session_id, phase, event_type) AS group_size
            FROM dedup
        """)
        con.execute("""
            CREATE TABLE pairs AS
            SELECT a.user_id, a.session_id AS group_id, a.phase,
                   CASE WHEN a.event_type = 'purchase' THEN 'buy' ELSE 'view' END AS relation,
                   a.item_id AS source_id, b.item_id AS target_id,
                   a.category AS source_category, b.category AS target_category
            FROM session_items a JOIN session_items b
              USING (user_id, session_id, phase, event_type)
            WHERE a.item_id < b.item_id
              AND a.category IS NOT NULL AND b.category IS NOT NULL
              AND a.group_size BETWEEN 2 AND (CASE WHEN a.event_type = 'purchase' THEN 30 ELSE 50 END)
              AND (a.phase = 'train' OR a.event_type = 'purchase')
        """)
        counts = dict(con.execute("SELECT phase, count(*) FROM pairs GROUP BY phase").fetchall())
        for phase in ("train", "validation", "test"):
            if not counts.get(phase):
                raise ValueError(f"Empty {phase} after applying cohort and session constraints")
            con.execute("""
                COPY (
                    SELECT user_id, group_id, source_id, target_id, relation, source_category, target_category
                    FROM pairs WHERE phase = ?
                    ORDER BY user_id, group_id, source_id, target_id, relation
                ) TO ? (FORMAT CSV, HEADER true, COMPRESSION gzip)
            """, [phase, str(output / f"{phase}.csv.gz")])
        users = con.execute("SELECT count(*) FROM cohort").fetchone()[0]
        crossing = con.execute("SELECT count(*) FROM sessions WHERE phase IS NULL").fetchone()[0]
    info = {
        "source": "REES46 Cosmetics", "validation_start": cuts[0], "test_start": cuts[1],
        "events_with_session": n_events, "crossing_sessions_removed": crossing,
        "max_buy_session_items": 30, "max_view_session_items": 50, "cohort_users": users,
        "cohort_selection": "Most training purchase sessions with 2–30 distinct products",
        "pair_rows": counts,
    }
    (output / "split_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    print(info, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=5000)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.database, args.max_users)
