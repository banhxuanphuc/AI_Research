from research_scientist.ingestion.grobid import parse_tei
from research_scientist.ingestion.scholarly import ArxivClient

TEI = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc><titleStmt><title level="a" type="main">A Tiny Paper</title></titleStmt></fileDesc>
    <profileDesc><abstract><div><p>We propose a tiny method.</p></div></abstract></profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head n="1">Introduction</head><p coords="1,10,10,100,20">Intro text.</p></div>
      <div><head n="2">Method</head>
        <p coords="2,10,10,100,20;2,10,40,100,20">Method text.</p>
        <figure type="table" coords="3,1,1,1,1"><figDesc>Accuracy on CIFAR-10.</figDesc></figure>
      </div>
    </body>
    <back><div><listBibl><biblStruct/><biblStruct/></listBibl></div></back>
  </text>
</TEI>"""

ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All
      You Need</title>
    <summary>The dominant sequence transduction models...</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related"/>
  </entry>
</feed>"""


def test_parse_tei_sections_pages_and_captions():
    doc = parse_tei(TEI, parser_version="0.8.1")
    assert doc.title == "A Tiny Paper"
    assert [s.path for s in doc.sections] == ["Abstract", "1 Introduction", "2 Method"]
    method = doc.sections[2]
    assert method.paragraphs[0].page == 2
    assert method.paragraphs[1].text == "[Table] Accuracy on CIFAR-10."
    assert method.paragraphs[1].page == 3
    assert doc.reference_count == 2


def test_parse_arxiv_feed():
    [paper] = ArxivClient.parse_feed(ATOM)
    assert paper.title == "Attention Is All You Need"
    assert paper.arxiv_id == "1706.03762"
    assert paper.year == 2017
    assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper.is_preprint is True
    assert paper.open_access_pdf.endswith("1706.03762v7")
