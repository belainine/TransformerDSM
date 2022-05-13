# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:47:58 2020

@author: belainine
"""
import xml.etree.ElementTree as ET
        

srcfile = """<srcset setid="newstest2015" srclang="any">
<doc sysid="ref" docid="1012-bbc" genre="news" origlang="en">
<p>
<seg id="1">India and Japan prime ministers meet in Tokyo</seg>
<seg id="2">India's new prime minister, Narendra Modi, is meeting his Japanese counterpart, Shinzo Abe, in Tokyo to discuss economic and security ties, on his first major foreign visit since winning May's election.</seg>
<seg id="3">Mr Modi is on a five-day trip to Japan to strengthen economic ties with the third largest economy in the world.</seg>
<seg id="4">High on the agenda are plans for greater nuclear co-operation.</seg>
<seg id="5">India is also reportedly hoping for a deal on defence collaboration between the two nations.</seg>
</p>
</doc>
<doc sysid="ref" docid="1018-lenta.ru" genre="news" origlang="ru">
<p>
<seg id="1">FANO Russia will hold a final Expert Session</seg>
<seg id="2">The Federal Agency of Scientific Organizations (FANO Russia), in joint cooperation with RAS, will hold the third Expert Session on “Evaluating the effectiveness of activities of scientific organizations”.</seg>
<seg id="3">The gathering will be the final one in a series of meetings held by the agency over the course of the year, reports a press release delivered to the editorial offices of Lenta.ru.</seg>
<seg id="4">At the third meeting, it is planned that the results of the work conducted by the Expert Session over the past year will be presented and that a final checklist to evaluate the effectiveness of scientific organizations will be developed.</seg>
</p>
</doc>
<srcset>"""

#ntok = NISTTokenizer()
src='data\dev\newstest2014-fren-ref.fr.sgm'
tree = ET.parse(src)
root = tree.getroot()
seg = root.findall('.//seg')
for n in seg:
   print(n.text)