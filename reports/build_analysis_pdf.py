"""Arma el PDF de la entrega. No forma parte del entrenamiento."""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
FIGURE = (
    ROOT
    / "reports"
    / "figures"
    / "confusion_matrix_c10.0_ngram1_balanced.png"
)
OUTPUT = ROOT / "reports" / "TC5061_Experimentacion_MLFlow_MendozaMoraJoseLuis.pdf"
ARIAL = Path(r"C:\Windows\Fonts\arial.ttf")
ARIAL_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")

BODY = (
    "La mejor configuracion es C=10, unigramas, class_weight=balanced "
    "y semilla 42. En la corrida auspicious-shark-466 el F1 de spam fue "
    "0.9453, por encima del baseline con C=1.0 (F1 0.9049). Precision y "
    "recall de spam quedaron en 0.9453. La exactitud fue 0.9864, frente "
    "al piso de 0.8760 de predecir siempre ham."
    "\n\n"
    "El hiperparametro que mas movio el F1 fue class_weight. Al quitar "
    "el balanceo, con el resto igual al baseline, el F1 bajo de 0.9049 "
    "a 0.8214 y el recall de spam de 0.9297 a 0.7188. El modelo deja de "
    "compensar que el spam es cerca del 12% de los mensajes: la precision "
    "sube a 0.9583, pero se escapa mucho spam. C tambien importa. Con "
    "C=0.1 el F1 baja a 0.8864 y con C=10 sube a 0.9453. Incluir bigramas "
    "mejora el baseline a 0.9231, un cambio menor que relajar la "
    "regularizacion."
    "\n\n"
    "Las limitaciones son un solo split 80/20, sin validacion cruzada, "
    "de modo que otra particion puede cambiar el orden. No se probo C=10 "
    "junto con bigramas. El modelo es una bolsa de palabras y no captura "
    "mensajes cuyo vocabulario no estaba en el entrenamiento. Cinco "
    "corridas no son una busqueda. La semilla 42 hace repetible cada "
    "corrida, pero no estima la varianza entre particiones."
)

COMMAND = (
    "Comando de la mejor corrida:\n"
    ".venv\\Scripts\\python -m src.train --C 10\n\n"
    "Repositorio:\n"
    "https://github.com/zzJLMM/"
    "TC5061_Experimentacion_MLFlow_MendozaMoraJoseLuis"
)


class Report(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", size=9)
        self.cell(0, 10, f"{self.page_no()}", align="C")


def write_paragraph(pdf: FPDF, text: str) -> None:
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 5.8, text)
    pdf.ln(2)


def main() -> None:
    pdf = Report(format="Letter")
    pdf.add_font("Arial", "", str(ARIAL))
    pdf.add_font("Arial", "B", str(ARIAL_BOLD))
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(25, 22, 25)
    pdf.add_page()

    pdf.set_font("Arial", "B", size=14)
    pdf.multi_cell(
        0,
        8,
        "TC5061 Experimentacion con MLflow\nMendoza Mora Jose Luis",
    )
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    write_paragraph(pdf, BODY)
    write_paragraph(pdf, COMMAND)

    pdf.add_page()
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 8, "Comparacion de las cinco corridas", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Arial", "B", size=10)
    headers = ["Corrida", "C", "N-gramas", "Peso", "F1 spam"]
    widths = [48, 18, 24, 28, 24]
    for header, width in zip(headers, widths):
        pdf.cell(width, 7, header, border=1)
    pdf.ln()
    pdf.set_font("Arial", size=10)
    rows = [
        ("auspicious-shark-466", "10", "1,1", "balanced", "0.9453"),
        ("masked-tern-717", "1", "1,2", "balanced", "0.9231"),
        ("shivering-shark-268", "1", "1,1", "balanced", "0.9049"),
        ("capricious-bass-750", "0.1", "1,1", "balanced", "0.8864"),
        ("unique-fly-720", "1", "1,1", "none", "0.8214"),
    ]
    for row in rows:
        for value, width in zip(row, widths):
            pdf.cell(width, 7, value, border=1)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(
        0,
        8,
        "Matriz de confusion de la mejor corrida",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.image(str(FIGURE), w=110)
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    write_paragraph(
        pdf,
        "Experimento TC5061 SMS Spam Hiperparametros en "
        "http://127.0.0.1:5000. La corrida auspicious-shark-466 "
        "quedo Finished, con el modelo sms_spam_model y la matriz "
        "de confusion como artefacto.",
    )
    pdf.output(str(OUTPUT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
