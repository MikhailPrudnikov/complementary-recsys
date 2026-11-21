"""Editable presentation with notes matching defense-notes.md."""
from pathlib import Path
import subprocess
from pptx import Presentation
from pptx.util import Pt
from pptx.oxml.xmlchemy import OxmlElement
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

D=Path(__file__).resolve().parent
import re
speech=(D/'defense-notes.md').read_text().split('## Ответы на вопросы')[0]
NOTES=[part.strip() for part in re.split(r'(?m)^## Слайд ',speech)[1:]]
assert len(NOTES)==7
W,H=1280,720
INK='000000'; ACCENT='000000'; LIGHT='E6E6E6'; PALE='F3F3F3'; WHITE='FFFFFF'; RULE='CCCCCC'
for n,f in [('Arial','Arial.ttf'),('Arial-Bold','Arial Bold.ttf')]:
 pdfmetrics.registerFont(TTFont(n,'/System/Library/Fonts/Supplemental/'+f))
prs=Presentation();prs.slide_width=Pt(W);prs.slide_height=Pt(H)
slide=None

def rect(x,y,w,h,fill=PALE,rounded=True):
 sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,Pt(x),Pt(y),Pt(w),Pt(h))
 if rounded: sh.adjustments[0]=0.07
 sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill);sh.line.fill.background()
 return sh

def text(t,x,y,w,size=32,bold=False,color=INK,center=False,h=None):
 lines=t.split('\n');height=h or len(lines)*size*1.22+8
 assert x>=0 and x+w<=W and y>=0 and y+height<=H,(t,'outside slide')
 for ln in lines:
  assert pdfmetrics.stringWidth(ln,'Arial-Bold' if bold else 'Arial',size)<w-3,(ln,'too wide',w)
 sh=slide.shapes.add_textbox(Pt(x),Pt(y),Pt(w),Pt(height))
 tf=sh.text_frame;tf.clear();tf.word_wrap=False;tf.auto_size=MSO_AUTO_SIZE.NONE
 tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.vertical_anchor=MSO_ANCHOR.TOP
 for i,ln in enumerate(lines):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
  p.text=ln;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
  p.line_spacing=1.12;p.space_after=Pt(0);p.space_before=Pt(0)
  p.alignment=PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
 return sh

def line(x1,y1,x2,y2,color=RULE,width=2):
 sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Pt(x1),Pt(y1),Pt(x2),Pt(y2))
 sh.line.color.rgb=RGBColor.from_string(color);sh.line.width=Pt(width)

def arrow(x,y,direction='down'):
 sh=slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW if direction=='down' else MSO_SHAPE.RIGHT_ARROW,Pt(x),Pt(y),Pt(22 if direction=='down' else 32),Pt(30 if direction=='down' else 22))
 sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(ACCENT);sh.line.fill.background()

def new(title):
 global slide
 slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor.from_string(WHITE)
 if title:
  rect(56,30,64,5,ACCENT,False)
  text(title,56,55,1168,46,True)
 slide.notes_slide.notes_text_frame.text=NOTES[len(prs.slides)-1]

new('')
text('Помогают ли просмотры\nрекомендовать дополнения?',80,225,1120,54,True,center=True)

new('Задача: найти дополнения к товару')
text('На входе — товар и категория; на выходе — 10 рекомендаций.',56,147,1168,32)
for x,w,label,value in [(56,350,'Исходный товар','Фотоаппарат'),(460,350,'Категория дополнения','Объективы'),(864,360,'Результат','10 рекомендаций')]:
 rect(x,265,w,140,LIGHT if x==864 else PALE)
 text(label,x+24,287,w-48,26)
 text(value,x+24,335,w-48,34,True,color=ACCENT if x==864 else INK)
text('+',420,315,32,34,True,center=True);arrow(820,328,'right')
text('Идея',56,459,120,32,True,color=ACCENT)
text('Просмотры могут помочь, когда покупок мало.',202,459,1022,34,True)
text('Риск',56,522,120,32,True,color=ACCENT)
text('Просмотры также связывают товары-заменители.',202,522,1022,32)
rect(56,604,1168,66,LIGHT)
text('Проверяем, когда совместное обучение помогает, а когда мешает.',80,622,1120,31,True)

new('Одна общая часть — две задачи')
text('Основа — двухбашенная модель из статьи Allegro',56,124,1168,30)
rect(330,192,620,115,LIGHT)
text('Название и категория товара',354,207,572,29,center=True)
text('Общий обработчик → вектор',354,253,572,32,True,ACCENT,True)
line(640,307,640,335,ACCENT);line(348,335,932,335,ACCENT)
line(348,335,348,352,ACCENT);line(932,335,932,352,ACCENT)
arrow(337,351);arrow(921,351)
for x,title,body in [(56,'Ветвь покупок','Предсказывает связи покупок'),(664,'Ветвь просмотров','Предсказывает связи просмотров')]:
 rect(x,386,560,116,PALE)
 text(title,x+24,402,512,34,True,center=True)
 text(body,x+24,452,512,29,center=True)
arrow(337,512)
text('10 дополнений к товару',80,552,512,32,True,ACCENT,True)
text('Обе задачи обучают общую часть.\nВ выдаче работает ветвь покупок.',664,530,560,30)
line(56,622,1224,622)
text('В каждой ветви — свои головы запроса и кандидата.',56,646,1168,30)

new('Два бейзлайна и четыре исследуемых метода')
text('Бейзлайны — простые ориентиры',56,132,1168,31,True,ACCENT)
for y,a,b in [(183,'MostPopular в категории','Популярные товары'),(245,'Только покупки','Нейросеть без просмотров')]:
 rect(56,y,1168,54,PALE)
 text(a,80,y+9,556,29);text(b,668,y+9,530,29)
text('Исследуемые методы',56,325,1168,31,True,ACCENT)
rows=[('Покупки + просмотры вперемешку','Нужно ли различать связи?'),('Общая часть, отдельные головы','Помогает ли совместное обучение?'),('Сначала просмотры, затем покупки','Достаточно ли предобучения?'),('Tucker 3D','Каковы качество и затраты?')]
for i,(a,b) in enumerate(rows):
 y=377+i*68
 rect(56,y,1168,58,LIGHT if i==1 else PALE)
 text(a,80,y+11,556,29,i==1,ACCENT if i==1 else INK)
 text(b,668,y+11,530,29)

new('Tucker 3D: другой способ объединить связи')
rect(56,141,1168,100,LIGHT)
text('Товар × товар × тип связи',80,162,1120,38,True,ACCENT,True)
text('Тип связи — просмотр или покупка',80,267,1120,32,center=True)
for x,w,title,body in [(56,350,'Векторы товаров','Общие для двух ролей'),(460,350,'Вектор типа связи','Покупка или просмотр'),(864,360,'Ядро','Связывает векторы')]:
 rect(x,340,w,126,PALE)
 text(title,x+20,357,w-40,30,True,center=True)
 text(body,x+20,411,w-40,26,center=True)
text('+',419,381,32,32,True,center=True);text('+',823,381,32,32,True,center=True)
text('Учимся на связях; полный тензор не храним.\nРанжируем дополнения по оценке связи «покупка».',56,503,1168,32)
rect(56,610,1168,75,LIGHT)
text('Гипотеза: Tucker 3D обходит оба простых бейзлайна.',80,632,1120,31,True)

new('Проверяем пользу при нехватке покупок')
text('Amazon Reviews 2018 · раздел Musical Instruments',56,124,1168,30,True)
text('Проверены 1000 строк; далее — проверка полного набора и шума связей.',56,173,1168,29)
rect(56,242,548,292,PALE)
text('Доля обучающих связей покупок',80,265,500,28,True)
for y,f,label in [(329,1,'100%'),(393,.5,'50%'),(457,.25,'25%')]:
 text(label,80,y+2,94,30,True)
 rect(191,y,380,39,'DDDDDD',False);rect(191,y,380*f,39,ACCENT,False)
text('Просмотры и тест фиксированы.\nТри запуска обучаемых моделей.',652,258,572,30,True)
text('NDCG@10',652,367,572,32,True,ACCENT)
text('Насколько удачен порядок выдачи',652,410,572,29)
text('Recall@10',652,464,572,32,True,ACCENT)
text('Какая доля нужных товаров найдена',652,507,572,28)
line(56,572,1224,572)
text('80 / 10 / 10 — обучение / проверка / тест. Одинаковые кандидаты.',56,597,1168,29)
text('Пары проверки и теста убираем из обоих обучающих источников.',56,643,1168,29)

new('Команда и этапы работы')
for x,name,body in [(56,'Данные и оценка','Данные и метрики\nMostPopular\nТолько покупки'),(456,'Двухбашенные модели','Общая часть модели\nСовместное обучение\nПредобучение'),(856,'Тензорные модели','Tucker 3D/4D и HOSVD\nСмешивание связей\nВремя и параметры')]:
 rect(x,138,368,199,PALE)
 text(name,x+24,158,320,27,True)
 text(body,x+24,216,320,28)
text('Запуски делим поровну. Каждый проверяет работу одного коллеги.',56,361,1168,29)
line(56,433,1224,433)
for x,date,label in [(56,'22.10','План проекта'),(294,'19.11','Первое\nсравнение'),(532,'03.12','Повторные\nэксперименты'),(770,'09.12','Финальный\nотчёт'),(1008,'10–11.12','Защита')]:
 rect(x,429,8,8,ACCENT,False)
 text(date,x,451,216,31,True,ACCENT)
 text(label,x,497,216,26)
text('Один GPU Kaggle; при нехватке ресурсов уменьшаем каталог и модели.\nРиски: шумные связи и лимиты GPU. Время оценим первыми запусками.',56,581,1168,27)
url='https://github.com/MikhailPrudnikov/complementary-recsys'
s=text('Репозиторий проекта',56,669,1168,24,color=ACCENT)
s.text_frame.paragraphs[0].runs[0].hyperlink.address=url
# Disable inherited theme shadows for every shape and connector.
for page in prs.slides:
 for shape in page.shapes:
  sp=shape._element.spPr
  sp.append(OxmlElement('a:effectLst'))
  for ref in shape._element.xpath('.//a:effectRef'): ref.set('idx','0')
# Keep hyperlinks in the same restrained accent colour.
from lxml import etree
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
for part in prs.part.package.iter_parts():
 if str(part.partname).startswith('/ppt/theme/') and str(part.partname).endswith('.xml'):
  root=etree.fromstring(part.blob)
  for tag in ['hlink','folHlink']:
   for node in root.xpath('//a:clrScheme/a:'+tag,namespaces=ns):
    for child in list(node): node.remove(child)
    c=etree.SubElement(node,'{'+ns['a']+'}srgbClr');c.set('val',ACCENT)
  part._blob=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
prs.save(D/'slides.pptx')
print('Built slides.pptx: 7 slides, speaker notes retained.')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--pdf',action='store_true');args=p.parse_args()
 if args.pdf:
  subprocess.run(['/Applications/LibreOffice.app/Contents/MacOS/soffice','-env:UserInstallation=file:///tmp/shared-lo-export','--headless','--convert-to','pdf','--outdir',str(D),str(D/'slides.pptx')],check=True)
