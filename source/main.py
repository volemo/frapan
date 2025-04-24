from pyscript import fetch, document, window, when
from pyscript.js_modules import file_io
import frapan

fileName = document.getElementById('fileRead');
files = file_io.newFile()

@when('change', '#fileRead')
async def on_read(event): 
    path = fileName.value
    print(path)
    fred = await files.read('fileRead')
    print('hi ', fred)

@when('click', '#local')
async def on_save(event):   
    await files.save('testing', 'fred.txt')
