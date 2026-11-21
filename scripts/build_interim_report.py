"""Render the team-format interim PDF from a completed local study."""

import argparse
from html import escape
import json
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle


INK = colors.HexColor("#182535")
MUTED = colors.HexColor("#47566B")
ACCENT = colors.HexColor("#1F5475")
REPO = "https://github.com/MikhailPrudnikov/complementary-recsys"


def fonts() -> None:
    pairs = [
        (Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
         Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in pairs:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("Text", str(regular)))
            pdfmetrics.registerFont(TTFont("TextBold", str(bold)))
            pdfmetrics.registerFontFamily("Text", normal="Text", bold="TextBold")
            return
    raise FileNotFoundError("Install Arial or DejaVu Sans with Cyrillic glyphs")


def footer(canvas, document) -> None:
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#D7DFE7"))
    canvas.line(47, 39, width - 47, 39)
    canvas.setFillColor(MUTED)
    canvas.setFont("Text", 8)
    canvas.drawString(47, 26, "Промежуточный отчёт")
    canvas.drawRightString(width - 47, 26, str(document.page))


def make_table(rows: list, widths: list[float]) -> Table:
    table = Table(rows, colWidths=[489 * width / sum(widths) for width in widths],
                  repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Text"),
        ("FONTNAME", (0, 0), (-1, 0), "TextBold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E4EDF3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#CBD7E3")),
    ]))
    return table


def sparseness_chart(summary: pd.DataFrame) -> Drawing:
    methods = ["Tucker", "Shared / Buy-only (история)", "Графовый метод"]
    comparisons = [
        ("Tucker buy+view", "Tucker buy-only"),
        ("TwoTower Shared", "TwoTower Buy-only"),
        ("Graph mixed paths", "Graph buy-buy"),
    ]
    fractions = [1.0, 0.5, 0.25]
    means = summary.pivot(index="buy_fraction", columns="model", values="ndcg_mean")
    values = [[means.at[fraction, first] - means.at[fraction, second] for fraction in fractions]
              for first, second in comparisons]
    scale = max(abs(value) for line in values for value in line) or 0.001
    chart = Drawing(489, 141)
    chart.add(String(0, 128, "Изменение NDCG@10 при добавлении просмотров",
                     fontName="TextBold", fontSize=10, fillColor=INK))
    palette = [ACCENT, colors.HexColor("#388879"), colors.HexColor("#A85B32")]
    center, available = 322, 115
    chart.add(Line(center, 22, center, 111, strokeColor=colors.HexColor("#C5CFD9"), strokeWidth=0.6))
    for index, (method, line) in enumerate(zip(methods, values)):
        y = 100 - index * 32
        chart.add(String(0, y - 5, method, fontName="Text", fontSize=8.5, fillColor=INK))
        for band, value in enumerate(line):
            length = value * available / scale
            chart.add(Rect(min(center, center + length), y - band * 8, abs(length), 5,
                           fillColor=palette[band], strokeColor=None))
    for band, text in enumerate(("100% покупок", "50%", "25%")):
        x = 8 + band * 105
        chart.add(Rect(x, 4, 9, 6, fillColor=palette[band], strokeColor=None))
        chart.add(String(x + 13, 3, text, fontName="Text", fontSize=8, fillColor=MUTED))
    chart.add(String(209, 17, "Ухудшение", fontName="Text", fontSize=8, fillColor=MUTED))
    chart.add(String(405, 17, "Улучшение", fontName="Text", fontSize=8, fillColor=MUTED))
    return chart


def make_report(amazon: Path, historical_amazon: Path, temporal: Path, output: Path, full_amazon: bool,
                amazon_raw_rows: int | None, transfer: Path | None, examples: Path | None) -> None:
    fonts()
    current = pd.read_csv(amazon / "test_summary.csv")
    historic = pd.read_csv(historical_amazon / "test_summary.csv")
    controls = {"MostPopular", "TwoTower Buy-only", "TwoTower Shared"}
    main = pd.concat([current, historic.loc[historic["model"].isin(controls)]], ignore_index=True)
    ranking_order = ["hit_rate_mean", "ndcg_mean", "recall_mean"]
    user_study = pd.read_csv(temporal / "test_summary.csv").sort_values(ranking_order, ascending=False)
    if len(pd.read_csv(amazon / "test_runs.csv")["seed"].unique()) != 3:
        raise ValueError("Expected three Amazon test seeds")
    catalog_size = json.loads((amazon / "test-100-seed-42" / "manifest.json").read_text())["catalog_size"]
    if full_amazon:
        provenance = "полный исходный файл метаданных Musical Instruments"
    elif amazon_raw_rows:
        provenance = f"первые {amazon_raw_rows:,} полных строк исходного файла".replace(",", " ")
    else:
        raise ValueError("Pass --amazon-raw-rows to identify a partial input")
    data_scope = f"{provenance}, {catalog_size:,} товаров в экспериментальном каталоге".replace(",", " ")
    paragraphs = {
        "title": ParagraphStyle("Title", fontName="TextBold", fontSize=21, leading=25,
                                alignment=TA_CENTER, textColor=INK, spaceAfter=12),
        "head": ParagraphStyle("Heading", fontName="TextBold", fontSize=13, leading=16,
                               textColor=ACCENT, spaceBefore=10, spaceAfter=5, keepWithNext=True),
        "text": ParagraphStyle("Text", fontName="Text", fontSize=10.1, leading=13.8,
                               alignment=TA_LEFT, textColor=INK, spaceAfter=5),
        "note": ParagraphStyle("Note", fontName="Text", fontSize=8.5, leading=12,
                               textColor=MUTED, spaceAfter=7),
        "reference": ParagraphStyle("Reference", fontName="Text", fontSize=8.5, leading=10.5,
                                    textColor=MUTED, spaceAfter=2),
    }
    story = []

    def add(text: str, style: str = "text") -> None:
        story.append(Paragraph(text, paragraphs[style]))

    add("Общие факторы для рекомендаций дополняющих товаров", "title")
    add("Введение", "head")
    add("Интернет-магазину нужно предложить к исходному товару подходящее дополнение, "
        "например чехол к гитаре. Совместные покупки дают полезный, но редкий сигнал. "
        "Совместные просмотры встречаются чаще и могут связывать как дополнения, так и заменители. "
        "Цель работы — измерить, помогают ли просмотры находить будущие совместные покупки "
        "и меняется ли их польза при уменьшении числа покупок.")
    add("Обзор литературы", "head")
    add("Двухбашенная модель Allegro [1] использует признаки товаров и целевую категорию. "
        "Она служит ориентиром для нашего поиска внутри категории. RESCAL [2] использует общие "
        "векторы объектов и матрицу для каждого отношения, а shared-factor Tucker [3] обобщает "
        "подход с общими факторами. HOSVD [4] даёт спектральное приближение разреженных таблиц. "
        "MMSC [5] использует типизированные пути в графе товаров. Из него мы заимствуем "
        "идею смешанных путей для простого графового контроля. В нашем эксперименте эффект "
        "просмотров изолирован внутри одной архитектуры, при общем каталоге и фиксированном тесте.")
    add("Постановка задачи", "head")
    add("Запрос состоит из исходного товара и заданной целевой категории. Модель возвращает "
        "до десяти товаров этой категории по убыванию балла совместной покупки. Правильными "
        "ответами считаем отложенные покупки из множества доступных кандидатов. В пилоте "
        "оцениваем товары, известные по train. Для персонализированной выдачи дополнительно "
        "используем известный идентификатор пользователя.")
    add("Предлагаемое решение", "head")
    add("Tucker 3D присваивает каждому товару один вектор u и учит отдельную матрицу M для "
        "каждого отношения. Балл пары — uᵢᵀ Mᵣ uⱼ. В режиме buy+view общий вектор "
        "обновляется по обеим задачам, при выдаче используем матрицу покупок. Tucker 4D "
        "добавляет фактор пользователя. HOSVD строит приближение по разреженным развёрткам. "
        "Градиентные модели используют BCE на наблюдаемых и выбранных ненаблюдаемых парах.")
    add("Mixed обучает общую пару голов запроса и кандидата на объединении покупок и просмотров. "
        "Для контролируемого сравнения локальные Buy-only, Mixed и Shared используют один "
        "энкодер с ID, категорией и хешированными токенами названия. У Shared отдельные головы "
        "для двух отношений и отдельный вес потерь на просмотрах. Графовый контроль оценивает пути из покупок и, во втором варианте, "
        "добавляет пути из покупки и просмотра.")
    add("Данные, гипотезы и экспериментальный сетап", "head")
    add(f"Основной источник — Amazon Reviews 2018, Musical Instruments [6]. В текущей серии — {data_scope}. "
        "Предобработка проекта также покрывает Retailrocket и REES46 Cosmetics/Electronics. "
        "Для REES46 храним пользователя, сессию и время и строим пары внутри сессий. "
        "На Cosmetics персонализированный эксперимент ограничен 3 000 пользователей с наибольшим "
        "числом подходящих покупочных сессий в train. Временные пороги строятся по всем "
        "событиям с сессиями, а пересекающие границы сессии исключаются. Ограничение длины "
        "сессии — 30 разных купленных товаров или 50 просмотренных.")
    story.append(make_table([
        ["Набор", "Товары в аудите", "Уникальные buy-пары", "Уникальные view-пары"],
        ["Amazon Musical Instruments", "112 135", "82 893", "227 621"],
        ["Retailrocket", "235 061", "7 052", "1 034 911"],
        ["REES46 Cosmetics", "54 571", "7 779 980", "13 113 113"],
        ["REES46 Electronics", "53 453", "4 053", "260 116"],
    ], [155, 103, 111, 120]))
    add("Подготовленные данные до выделения экспериментальных каталогов и когорт.", "note")
    add("Amazon делим примерно на 80/10/10 по неориентированному ключу пары. Отложенные пары "
        "исключены из обучающих покупок и просмотров. В исправленной серии отрицательные примеры "
        "выбираем по доступным связям train. Mixed заново выбирает просмотры каждую эпоху. "
        "Исторические Buy-only и Shared использовали список отложенных пар для исключения их из "
        "отрицательных примеров. Доли покупок 100%, 50%, 25% вложены. Каталог, тест и кандидаты фиксированы. "
        "Ранги и веса взяты из первоначальной настройки на validation. В таблицах показываем "
        "среднее и стандартное отклонение Recall@10, NDCG@10 и HitRate@10 при seed 42, 43 и 44. "
        "HitRate@10 — доля запросов с хотя бы одним релевантным товаром в топ-10. Время измеряем на CPU.")
    add("Текущие результаты и выводы", "head")
    rows = [["Модель", "Recall@10", "NDCG@10", "HitRate@10", "Параметры", "Обуч., с", "Оценка, с"]]
    full = main.loc[main["buy_fraction"].eq(1)].sort_values(ranking_order, ascending=False).set_index("model")
    report_models = {"MostPopular", "Tucker buy-only", "Tucker buy+view", "TwoTower Buy-only",
                     "TwoTower Mixed", "TwoTower Shared", "HOSVD", "DistMult buy+view",
                     "Graph mixed paths", "MIG-TF"}
    for name, result in full.loc[full.index.isin(report_models)].iterrows():
        label = f"{name}*" if name in controls else name
        rows.append([label, f"{result.recall_mean:.3f} ± {result.recall_std:.3f}",
                     f"{result.ndcg_mean:.3f} ± {result.ndcg_std:.3f}",
                     f"{result.hit_rate_mean:.3f} ± {result.hit_rate_std:.3f}",
                     f"{int(result.parameters):,}".replace(",", " "),
                     f"{result.fit_seconds:.2f}", f"{result.evaluation_seconds:.2f}"])
    story.append(make_table(rows, [113, 67, 67, 67, 62, 55, 58]))
    add(f"Amazon test, 100% обучающих покупок, {int(full.iloc[0].queries):,} запросов. "
        "Время — медиана трёх запусков, время оценки включает построение выдачи и расчёт метрик. "
        "Порядок строк: HitRate@10, затем NDCG@10, затем Recall@10. Для HOSVD указано число элементов факторов и ядра.", "note")
    add("* Результаты первоначальной серии на тех же тестовых запросах.", "note")
    story.append(sparseness_chart(main))
    add("Столбцы показывают разницу между режимом с просмотрами и контрольным режимом "
        "того же семейства. Вертикальная линия соответствует нулевой разнице.", "note")

    best = full["ndcg_mean"].idxmax()
    tucker_delta = full.at["Tucker buy+view", "ndcg_mean"] - full.at["Tucker buy-only", "ndcg_mean"]
    graph_delta = full.at["Graph mixed paths", "ndcg_mean"] - full.at["Graph buy-buy", "ndcg_mean"]
    add(f"На этом Amazon split при 100% покупок наибольший средний NDCG@10 показывает "
        f"{escape(best)} ({full.at[best, 'ndcg_mean']:.3f}). Добавление просмотров меняет "
        f"NDCG@10 Tucker на {tucker_delta:+.3f}, графового контроля на {graph_delta:+.3f}. "
        "Для Mixed и исторического Shared различается также протокол обучения. "
        "По их сравнению нельзя оценить вклад раздельных голов.")
    tucker = full.loc["Tucker buy+view"]
    shared = full.loc["TwoTower Shared"]
    recall_delta = tucker.recall_mean - shared.recall_mean
    ndcg_delta = tucker.ndcg_mean - shared.ndcg_mean
    conclusion = ("По точечным оценкам Tucker не уступает Shared более чем на 0,01 ни по одной метрике."
                  if min(recall_delta, ndcg_delta) >= -0.01
                  else "По крайней мере по одной метрике Tucker уступает Shared более чем на 0,01.")
    add(f"По сравнению с историческим Shared разница Tucker по Recall@10 равна {recall_delta:+.3f}, "
        f"по NDCG@10 — {ndcg_delta:+.3f}. {conclusion} "
        f"Число параметров Tucker составляет {tucker.parameters / shared.parameters:.1%} от Shared.")
    rows_4d = [["Модель", "Recall@10", "NDCG@10", "HitRate@10", "Параметры", "Запросы"]]
    for row in user_study.itertuples(index=False):
        rows_4d.append([row.model, f"{row.recall_mean:.3f} ± {row.recall_std:.3f}",
                        f"{row.ndcg_mean:.3f} ± {row.ndcg_std:.3f}",
                        f"{row.hit_rate_mean:.3f} ± {row.hit_rate_std:.3f}",
                        f"{int(row.parameters):,}".replace(",", " "), str(int(row.queries))])
    story.append(make_table(rows_4d, [126, 76, 79, 79, 69, 60]))
    add(f"REES46 Cosmetics, общий test для 3D и 4D. Покрытие после фильтров "
        f"{user_study.iloc[0].coverage:.1%} среди запросов выбранной когорты. "
        "Выводы о персонализации относятся к активным пользователям этой когорты.", "note")
    rows_4d_by_name = user_study.set_index("model")
    personalization_delta = (rows_4d_by_name.at["Tucker 4D", "ndcg_mean"]
                             - rows_4d_by_name.at["Tucker 3D", "ndcg_mean"])
    add(f"Добавление пользователя изменило средний NDCG@10 на {personalization_delta:+.3f}. "
        f"Разброс 4D по инициализациям равен "
        f"{rows_4d_by_name.at['Tucker 4D', 'ndcg_std']:.3f}.")
    if transfer is not None:
        extra = pd.read_csv(transfer / "test_summary.csv")
        extra = extra.loc[extra["buy_fraction"].eq(1)].set_index("model")
        leader = extra["ndcg_mean"].idxmax()
        gain = extra.at["Tucker buy+view", "ndcg_mean"] - extra.at["Tucker buy-only", "ndcg_mean"]
        add(f"На контрольном временном split REES46 Electronics "
            f"({int(extra.iloc[0].queries)} test-запросов) лидировал {escape(leader)} "
            f"с NDCG@10 {extra.at[leader, 'ndcg_mean']:.3f}. Внутри Tucker эффект просмотров "
            f"составил {gain:+.3f}.")
    if examples is not None:
        for case in json.loads(examples.read_text()):
            add(escape(case["text"]), "note")
    add("Дальше проверим устойчивость результата на других каталогах, расширим ручную оценку "
        "совместимости и подключим командные варианты Pretrain и Transformer.")
    add("Воспроизведение", "head")
    add(f"Репозиторий команды — <link href='{REPO}' color='#1F5475'>{REPO}</link>. "
        "Установите пакет через <b>python -m pip install -e .</b>. В README приведены источники "
        "данных, команды подготовки сплитов Amazon и REES46 и команды запуска. Для исправленной "
        "серии Amazon используется configs/filipp_owned.json с --fixed-settings. Таблицы, "
        "использованные при сборке PDF, лежат в docs/results.")
    add("Источники", "head")
    for number, title, link in [
        (1, "Allegro, Complementary Two Tower, RecSys 2025", "https://arxiv.org/abs/2508.03702"),
        (2, "Nickel et al., RESCAL, ICML 2011", "https://icml.cc/2011/papers/438_icmlpaper.pdf"),
        (3, "Peshekhonov et al., Shared-factor Tucker, AISTATS 2024", "https://proceedings.mlr.press/v238/peshekhonov24a.html"),
        (4, "De Lathauwer et al., HOSVD, 2000", "https://doi.org/10.1137/S0895479896305696"),
        (5, "Wang et al., MMSC, SIGIR 2026", "https://arxiv.org/abs/2507.22268"),
        (6, "Ni et al., Amazon Reviews 2018", "https://nijianmo.github.io/amazon/index.html"),
        (7, "REES46 eCommerce Events History in Cosmetics Shop", "https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop"),
    ]:
        add(f"[{number}] <link href='{link}' color='#1F5475'>{escape(title)}</link>", "reference")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=47, rightMargin=47,
                               topMargin=42, bottomMargin=51,
                               title="Общие факторы для рекомендаций дополняющих товаров")
    report.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amazon", type=Path, required=True)
    parser.add_argument("--historical-amazon", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-amazon", action="store_true")
    parser.add_argument("--amazon-raw-rows", type=int)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--examples", type=Path)
    args = parser.parse_args()
    make_report(args.amazon, args.historical_amazon, args.temporal, args.output, args.full_amazon,
                args.amazon_raw_rows, args.transfer, args.examples)
