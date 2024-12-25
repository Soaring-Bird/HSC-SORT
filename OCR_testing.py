import http.client
import base64

# Function to perform OCR on a given PDF file
def ocr_pdf(file_path):
    # Read the PDF file as binary
    with open(file_path, "rb") as file:
        encoded_file = base64.b64encode(file.read()).decode('utf-8')
    
    # Define payload with the encoded file
    payload = f"file={encoded_file}&isOverlayRequired=false&filetype=pdf"
    
    # Define headers
    headers = {
        'x-rapidapi-key': "0c6806267dmsh7b55d1faf943d4ep1c8344jsn1d1c95a0376e",
        'x-rapidapi-host': "ocr-text-extraction.p.rapidapi.com",
        'Content-Type': "application/x-www-form-urlencoded"
    }
    
    # Create HTTPS connection and make the request
    conn = http.client.HTTPSConnection("ocr-text-extraction.p.rapidapi.com")
    conn.request("POST", "/v1/ocr/", payload, headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    
    # Parse and return the OCR result
    result = data.decode("utf-8")
    return result

# Specify the path to your local PDF file
file_path = ""Ruse Bio 2020.pdf""

# Perform OCR and print the result
try:
    ocr_result = ocr_pdf(file_path)
    print("OCR Text Output:")
    print(ocr_result)
except Exception as e:
    print("An error occurred:", e)
