"""Current two-page proposal: shared representations for purchase/view relations."""
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parent
FD=Path('/System/Library/Fonts/Supplemental')
for name,file in [('Arial','Arial.ttf'),('Arial-Bold','Arial Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FD/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold')
INK='#000000'; TEAL='#000000'; GREY='#000000'; PALE='#FFFFFF'
S={
 'title':ParagraphStyle('title',fontName='Arial-Bold',fontSize=21,leading=25,textColor=colors.HexColor(INK),spaceAfter=9),
 'body':ParagraphStyle('body',fontName='Arial',fontSize=10,leading=13.8,spaceAfter=6,textColor=colors.HexColor(INK)),
 'h':ParagraphStyle('h',fontName='Arial-Bold',fontSize=12,leading=15,spaceBefore=12,spaceAfter=4,textColor=colors.HexColor(TEAL)),
 'prompt':ParagraphStyle('prompt',fontName='Arial',fontSize=8.6,leading=11.3,spaceAfter=6,textColor=colors.HexColor(GREY)),
 'small':ParagraphStyle('small',fontName='Arial',fontSize=8.1,leading=10.5,spaceAfter=3,textColor=colors.HexColor(GREY)),
 'bullet':ParagraphStyle('bullet',fontName='Arial',fontSize=10,leading=13.8,leftIndent=12,firstLineIndent=0,bulletIndent=0,spaceAfter=4,textColor=colors.black),
 'cell':ParagraphStyle('cell',fontName='Arial',fontSize=9,leading=12,textColor=colors.HexColor(INK)),
}
def p(t,style='body'):return Paragraph(t,S[style])
def section(n,title,prompt):return [p(f'{n}. {title}','h')]
def table(rows,widths):
    t=Table([[p(c,'cell') for c in row] for row in rows],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor(PALE)),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,-1),(-1,-1),.5,colors.black)]))
    return t

def footer(c,doc):
    w,h=A4
    c.setFont('Arial',8);c.setFillColor(colors.HexColor(GREY))
    pass

story=[p('Помогают ли просмотры<br/>рекомендовать дополнения?','title')]
story+=section(1,'Задача и цель проекта','')
story += [p('На входе — товар и категория дополнения, например <b>«фотоаппарат + объективы»</b>. На выходе — десять товаров этой категории. Исследуем, помогают ли совместные просмотры находить дополнения, особенно когда связей покупок мало.'),p('<b>Проверяемые гипотезы:</b>'),Paragraph('<b>Польза просмотров.</b> Совместное обучение с отдельными головами превосходит модель «только покупки».',S['bullet'],bulletText='•'),Paragraph('<b>Разделение связей.</b> Отдельные головы дают лучшее качество, чем смешивание покупок и просмотров.',S['bullet'],bulletText='•'),Paragraph('<b>Одновременное обучение.</b> Совместное обучение лучше схемы «сначала просмотры, затем покупки»: одного предобучения недостаточно.',S['bullet'],bulletText='•'),Paragraph('<b>Тензорный подход.</b> Tucker 3D, обученный на покупках и просмотрах, превосходит MostPopular и модель «только покупки».',S['bullet'],bulletText='•'),Paragraph('<b>Нехватка покупок.</b> При уменьшении доли покупок со 100% до 25% преимущество совместной нейросети и Tucker 3D над моделью «только покупки» растёт.',S['bullet'],bulletText='•')]
story+=section(2,'Существующие решения и наша идея','')
story += [p('<b>Основа — <link href="https://arxiv.org/abs/2508.03702" color="#000000">Suggest, complement, inspire: Two Tower в Allegro</link>.</b> Берём сценарий поиска дополнений по товару и целевой категории. Реализуем упрощённую модель на открытых данных.'),p('<b>Самостоятельная работа:</b> сравниваем способы использования просмотров в двухбашенной модели и Tucker 3D на одних товарных связях. Проверяем гипотезы выше; между совместной нейросетью и Tucker сравниваем качество, число параметров и время, не предполагая победителя заранее.')]
story+=section(3,'Предлагаемый метод','')
story += [p('<b>Двухбашенная модель.</b> Общий обработчик превращает название и категорию товара в вектор. Отдельные головы запроса и кандидата оценивают покупки и просмотры. Запрос покупок учитывает категорию дополнения; при выдаче используем только эту ветвь.'),p('Учимся отличать известные связи от выбранных ненаблюдаемых пар. Для совместного обучения: <b>L = L<sub>покупки</sub> + λ · L<sub>просмотры</sub></b>. Вес λ подбираем на проверочной выборке.'),p('<b>MostPopular:</b> ранжируем кандидатов нужной категории по числу разных товаров, связанных с ними покупкой в обучении. Исходный товар на оценку не влияет; кандидаты и исключения те же. При равных оценках — порядок по идентификатору.'),p('<b>3D-тензор: товар × товар × тип связи.</b> Третья координата — «просмотр» или «покупка». Единица обозначает наблюдаемую связь. Модель совместно изучает оба отношения; кандидатов ранжируем по оценке связи «покупка».'),p('<b>Обучение 3D.</b> Используем <link href="https://aclanthology.org/D19-1522/" color="#000000">разложение Tucker</link>: общую таблицу векторов для двух товарных осей, факторы типов связи и небольшое ядро взаимодействий. Обучаем по бинарной кросс-энтропии на наблюдаемых и выбранных ненаблюдаемых связях. Полный тензор не создаём.'),p('Оцениваем товары, встречавшиеся в обучении. Tucker 3D использует идентификаторы, нейросети — название и категорию. Поэтому сравнение этих моделей показывает качество подходов целиком, а не только влияние архитектуры.')]
story.append(PageBreak())
story+=section(4,'Данные и протокол оценки','')
story += [p('<b><link href="https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/" color="#000000">Amazon Reviews 2018</link></b>: названия, категории, also_buy и also_view; начнём с Musical Instruments. Поля проверены на первых 1000 строках; далее проверим полноту связей раздела.'),p('Разбиение обучение / проверка / тест ≈ <b>80 / 10 / 10</b>. Пару и обратное направление держим вместе; пары проверки и теста исключаем из обоих источников обучения. Кандидаты и исключение известных покупок одинаковы.'),p('<b>NDCG@10</b> оценивает порядок дополнений, <b>Recall@10</b> — долю найденных. Связи покупок — приближённая разметка: отбираем пары категорий и вручную разбираем примеры. Отсутствие связи не считаем доказанной несовместимостью. Учитываем время и число параметров.')]
story+=section(5,'Бейзлайны и исследуемые методы','')
story += [p('<b>Бейзлайны — два простых ориентира.</b> MostPopular рекомендует популярные товары нужной категории. «Только покупки» — исходная двухбашенная модель без просмотров.'),p('<b>Исследуемые методы:</b>'),table([['<b>Метод</b>','<b>Что выясняем сравнением</b>'],['Покупки и просмотры вперемешку','Достаточно ли добавить просмотры, не различая типы связей?'],['Общая часть, отдельные головы','Помогает ли совместное обучение с разделением связей? Это наш основной нейросетевой вариант.'],['Сначала просмотры, затем покупки','Достаточно ли предобучения на просмотрах вместо одновременного обучения?'],['Tucker 3D','Помогает ли тензорный подход обойти простые ориентиры? Каковы его качество и затраты относительно совместной нейросети?']],[215,300]),Spacer(1,6),p('Все <b>шесть методов</b> сравниваем при <b>100%, 50% и 25%</b> обучающих связей покупок. Подвыборки вложены и одинаковы для всех методов. Популярность пересчитываем на каждой подвыборке. Просмотры и тест фиксируем; настройки выбираем на проверке.'),p('Обучаемые модели запускаем с тремя инициализациями при сопоставимом бюджете настройки; показываем среднее и разброс NDCG@10 и Recall@10. MostPopular детерминирован. Для каждой гипотезы сравниваем соответствующие методы; отрицательный результат тоже фиксируем.')]
story+=section(6,'Команда, сроки и ресурсы','')
story += [table([['<b>Направление</b>','<b>Зона работы</b>'],['Данные и оценка','Данные, разбиение, метрики и два бейзлайна: MostPopular и «только покупки».'],['Двухбашенные модели','Общий обработчик и обучение; модели с отдельными головами и «сначала просмотры, затем покупки».'],['Тензорные модели','Tucker 3D/4D, HOSVD, смешивание покупок с просмотрами и сравнение затрат.']],[100,415]),Spacer(1,6),p('<b>Вместе:</b> запуски делим поровну; каждый анализирует свои модели и пишет часть отчёта. Каждый проверяет код и результаты одного коллеги. После первых запусков оцениваем объём оставшейся работы и при необходимости перераспределяем задачи.'),p('<b>22.10</b> — proposal; <b>19.11</b> — первое сравнение шести методов; <b>03.12</b> — нехватка покупок и повторные запуски; <b>09.12</b> — отчёт; <b>10–11.12</b> — защита.'),p('<b>Ресурсы:</b> один GPU Kaggle, до 20 тыс. товаров; время оценим пилотом. Риски: шумные или редкие связи, квоты GPU. При нехватке ресурсов уменьшаем каталог и модели, сохраняя шесть методов и единый протокол.'),p('<b>Репозиторий:</b> <link href="https://github.com/MikhailPrudnikov/complementary-recsys" color="#000000">код и инструкции на GitHub</link>','small')]
doc=SimpleDocTemplate(str(ROOT/'proposal-shared-learning.pdf'),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=30,bottomMargin=45,title='Общие представления для рекомендаций дополняющих товаров',author='Команда complementary-recsys')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(ROOT/'proposal-shared-learning.pdf')
