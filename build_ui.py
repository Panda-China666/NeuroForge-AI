from pathlib import Path
root=Path(__file__).resolve().parent/'static'
(root/'app.js').write_text((root/'base.js').read_text(encoding='utf-8')+'\n'+(root/'vnext.js').read_text(encoding='utf-8')+'\n'+(root/'studio.js').read_text(encoding='utf-8')+'\n'+(root/'agent_ui.js').read_text(encoding='utf-8-sig')+'\ninit();\n',encoding='utf-8')
print('Bundled base.js + vnext.js + studio.js + agent_ui.js -> app.js')
