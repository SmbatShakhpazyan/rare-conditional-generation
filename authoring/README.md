# Rebuild the written artifacts

Install Python requirements here and the Liberation Serif and DejaVu Math TeX
Gyre fonts. From repository root:

```sh
python authoring/author_full_document.py --content authoring/full_monograph_content.json --output work/rebuilt.docx
python authoring/export_markdown.py --content authoring/full_monograph_content.json --output paper/full_research_monograph_en.md
```

Displayed Word equations are native OMML. The seven-equation XML library is
included locally; the builder does not need the former Armenian document.
Font/renderer changes can change pagination. Contents page numbers in this
release are checked against the delivered PDF; regenerate them after edits.
Vector figure sources remain editable. To regenerate scientific figures:

```sh
python authoring/create_visuals.py
npm install --prefix authoring
node authoring/rasterize_figures.cjs
python authoring/create_poster.py
```

Rasterize the poster SVG with sharp or another SVG renderer. The precise
benchmark figures read the saved aggregate table; analytical figures are labeled
as constructed examples and do not simulate efficacy measurements.
