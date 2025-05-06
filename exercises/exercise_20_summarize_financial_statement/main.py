from pathlib import Path
import streamlit as st
import os
import tempfile
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image

def run():
    SummarizeFinancialStatement().run()

class SummarizeFinancialStatement:
    def run(self):
        st.title("📊 Financial Statement Summarizer")
        # File uploader for PDF
        uploaded_pdf = st.file_uploader("Upload a financial statement PDF file", type=["pdf"])
        if uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_pdf.read())
                temp_pdf_path = temp_pdf.name
            if st.button("Summarize"):
                image_paths = self.pdf_to_images(temp_pdf_path)
                response_text = self.generate_summary_report(
                    image_paths, 
                    prompt=self.prompt()
                )
                st.write(response_text)
    
    def pdf_to_images(self, pdf_path, output_folder="pdf_images"):
        # Create an ./images folder if not exists
        script_dir = Path(__file__).parent.resolve()
        image_folder = script_dir / output_folder   
        image_folder.mkdir(exist_ok=True)
        
        images = convert_from_path(pdf_path)
        image_paths = []

        for i, image in enumerate(images):
            image_path = os.path.join(image_folder, f"page_{i+1}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)

        return image_paths
    
    def generate_summary_report(self, image_paths, prompt):
        genai.configure(api_key="")

        #Load all images
        pdf_images = []
        for img_path in image_paths:
            pdf_images.append(Image.open(img_path))
        
        # Gemini expects list of [image, image, prompt...]
        gemini_inputs = pdf_images + [prompt]

        # Get the summary
        generation_config = genai.types.GenerationConfig(
            temperature=0
        )
        gemini = genai.GenerativeModel(
            model_name='gemini-1.5-flash-latest',
            generation_config=generation_config
        )
        response = gemini.generate_content(gemini_inputs)
        return response.text
    
    def prompt(self):
        return """
        You are an AI assistant specialized in analyzing financial reports and images. Your task is to extract key insights from financial document images, summarize complex financial content, and provide actionable insights tailored to the user's requirements.

        For each image:
        1. Identify the type of financial document or chart (e.g., balance sheet, income statement, cash flow statement, trend graph).
        2. Extract key financial metrics, trends, and data points visible in the image.
        3. Note any relevant textual information, headers, or labels that provide context.

        Based on the image analysis and the user's prompt, prepare a comprehensive financial report analysis. Follow these guidelines:

        1. Highlight key performance metrics such as revenue, net income, earnings per share (EPS), and growth trends compared to previous periods.
        2. Summarize important financial statements (balance sheet, income statement, cash flow statement) to reflect the company's assets, liabilities, equity, revenue, expenses, and cash flows.
        3. Provide insights into operational performance, market conditions, and management's commentary on future outlooks, risks, and strategic priorities.
        4. Include relevant financial ratios (e.g., profitability, liquidity, and leverage ratios) and any notable events such as acquisitions or major partnerships.
        5. Discuss risks, challenges, and external factors that might impact the business, along with the company's approach to managing those risks.
        6. If applicable, include shareholder-related information such as dividends or stock performance.

        Structure your output as follows:
        1. Summary of key financial metrics
        2. Income and expenses analysis
        3. Balance sheet highlights
        4. Credit quality assessment (if applicable)
        5. Strategic and operational updates
        6. Market conditions and outlook
        7. Conclusion

        Ensure your analysis is tailored to the user's specific request while providing a comprehensive overview of the company's financial health and strategic direction.

        In your final output, include only the financial report analysis based on the image analysis and user prompt. Do not include any explanations of your process or repetitions of the instructions. Present your analysis in a clear, concise manner suitable for financial professionals, investors, or decision-makers.

        Can we consider buying this company stock. Just provide Yes or No. Provide the reason if yes or No
        """