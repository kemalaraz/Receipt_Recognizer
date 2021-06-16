# Receipt_Recognizer
A library for detecting restaurant name, date and overall bill from a given receipt.

## For OCR Tesseract to work
### On Linux
sudo apt-get update
sudo apt-get install libleptonica-dev
sudo apt-get install tesseract-ocr tesseract-ocr-dev
sudo apt-get install libtesseract-dev
### On Mac
brew install tesseract
### On Windows
download binary from https://github.com/UB-Mannheim/tesseract/wiki. then add pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe' to your script.