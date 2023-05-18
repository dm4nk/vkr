import re

import aspose.words as aw
import pandas as pd

doc = aw.Document()
builder = aw.DocumentBuilder(doc)
table = builder.start_table()

builder.paragraph_format.alignment = aw.ParagraphAlignment.CENTER
builder.font.size = 12
builder.font.name = "Times New Roman"

df = pd.read_csv('results/linear_svm_semantic_tri.csv')

intro = ['N/A', 'N/A', 'лайки', 'комментарии', 'просмотры', 'репосты']

rows = ['семантика']

rows_titles = ['F1', 'Precision', 'Recall', 'Accuracy']

for m in intro:
    builder.insert_cell()
    builder.write(m)
builder.end_row()

w, h = 6, 4 * 1 + 1
results = [['N/A' for x in range(w)] for y in range(h)]
print(results)

for i in range(1, len(df)):
    for j in range(2, 3):
        numbers = re.findall('\d+\.\d+', df.iloc[i, j])[1:]

        for k in range(len(numbers)):
            results[(j - 2) * 4 + k][0] = rows[(j - 2)]
            results[(j - 2) * 4 + k][1] = rows_titles[k]
            results[(j - 2) * 4 + k][i + 1] = numbers[k]

print(results)

for r in results:
    for a in r:
        builder.insert_cell()
        builder.write(a)
    builder.end_row()

builder.end_table()
doc.save("table_formatted.docx")
