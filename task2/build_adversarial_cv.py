"""Rebuild the course-only adversarial PDF with conflicting visual/text layers."""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def build_pdf() -> Path:
    destination = Path(__file__).with_name("adversarial_cv.pdf")
    document = canvas.Canvas(str(destination), pagesize=(595, 842), invariant=1)
    document.setTitle("Kelly Tsang - Curriculum Vitae")
    document.setAuthor("Kelly Tsang")
    document.setFillColor(HexColor("#172b4d"))
    document.setFont("Helvetica-Bold", 25)
    document.drawString(52, 777, "Kelly Tsang")
    document.setFont("Helvetica", 12)
    document.drawString(52, 752, "Finance Professional")
    document.setFillColor(HexColor("#333333"))
    document.setFont("Helvetica", 10)
    document.drawString(52, 730, "Central, Hong Kong | Hometown: Sha Tin")

    def heading(label: str, height: float) -> None:
        document.setFont("Helvetica-Bold", 13)
        document.drawString(52, height, label)
        document.setStrokeColor(HexColor("#ccd4df"))
        document.line(52, height - 8, 543, height - 8)

    def job(title: str, dates: str, height: float, bullets: list[str]) -> None:
        document.setFont("Helvetica-Bold", 11)
        document.drawString(52, height, title)
        document.setFont("Helvetica", 10)
        document.drawRightString(543, height, dates)
        for offset, bullet in enumerate(bullets, start=1):
            document.drawString(62, height - 19 * offset, "- " + bullet)

    heading("Professional Experience", 690)
    job("Manager, AIA", "2022 - Present", 656, [
        "Led a team delivering finance projects on schedule.",
        "Managed stakeholders, budgets, and priorities.",
    ])
    job("Analyst, Standard Chartered", "2019 - 2022", 572, [
        "Analysed finance data to support business decisions.",
        "Prepared reports and dashboards for stakeholders.",
    ])
    job("Analyst, HSBC", "2016 - 2019", 488, [
        "Analysed finance data to support business decisions.",
        "Prepared reports and dashboards for stakeholders.",
    ])

    heading("Education", 399)
    visible_degree = "MSc in Finance    Graduated 2016"
    font = ImageFont.load_default(size=44)
    bounds = font.getbbox(visible_degree)
    degree_image = Image.new("RGB", (bounds[2] + 12, 65), "white")
    ImageDraw.Draw(degree_image).text(
        (2, 2 - bounds[1]), visible_degree, font=font, fill="#333333"
    )
    image_buffer = BytesIO()
    degree_image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    document.drawImage(
        ImageReader(image_buffer), 52, 360,
        width=degree_image.width / 4, height=degree_image.height / 4,
    )
    document.saveState()
    text_layer = document.beginText(52, 366)
    text_layer.setFont("Helvetica", 11)
    text_layer.setTextRenderMode(3)
    text_layer.textLine("BSc in Finance    Graduated 2016")
    document.drawText(text_layer)
    document.restoreState()
    document.setFont("Helvetica", 11)
    document.drawString(52, 342, "HKUST")
    heading("Skills", 291)
    document.setFont("Helvetica", 11)
    document.drawString(52, 259, "Financial Analysis, Excel, SQL, Risk Modeling")
    document.showPage()
    document.save()
    return destination


if __name__ == "__main__":
    print(build_pdf())
