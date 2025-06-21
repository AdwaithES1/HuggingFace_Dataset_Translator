# Dataset translator using IndicTrans2 model

This is a pipeline which uses IndicTrans2 model to translate datasets imported directly from HuggingFace into 22 indic languages {Assamese,Bengali,Bodo,Dogri,English,Gujarati,Hindi,Kannada,
Kashmiri (Arabic script and Devanagari script),Konkani,Maithili,Malayalam,Manipuri (Bengali script and Meitei script),Marathi,Nepali,Odia,Punjabi,Sanskrit,Santali (Ol Chiki script),Sindhi 
(Arabic script and Devanagari script),Tamil,Telugu,Urdu }.

Note : CUDA installation is a prerequisite, if not, the code will run on CPU which will be slower. Also don't forget to make sure CUDA and pytorch versions are compatible with eachother.

#To setup anaconda env

Run the commands:

conda env create -f environment.yml
conda activate dataset-translator

After it, you can run the translate.py program.

# Recommended batch sizes 
Vram          Batchsize

 4 GB            2

 6 GB           2-4

 8GB            4-8

 16 GB          8-16

 Note: If more space seems to be free from the VRAM, you can try increasing the batch size.
