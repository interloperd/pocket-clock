"""Intègre dashboard_data.json dans le template et produit la page autonome index.html."""
html = open("dashboard_template.html", encoding="utf-8").read()
data = open("dashboard_data.json", encoding="utf-8").read().replace("NaN", "null")
page = html.replace("/*DATA*/null", data)
# En-tête de page complet : indispensable pour que le téléphone affiche la version mobile
head = ('<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>html,body{margin:0}body{font-size:14px}img{max-width:100%}[hidden]{display:none!important}'
        ':root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}</style>\n')
i = page.index('<script>')
standalone = head + page[:i].replace('<title>', '<title>', 1) + '</head>\n<body>\n' if False else None
# le contenu du template commence par <title> et <style> : on les garde dans <head>, le reste va dans <body>
j = page.index('</style>', page.index('<style>')) + len('</style>')
standalone = head + page[:j] + '\n</head>\n<body>\n' + page[j:] + '\n</body>\n</html>\n'
for f in ("index.html", "pocket_clock_dashboard.html"):
    open(f, "w", encoding="utf-8").write(standalone)
open("artifact_page.html", "w", encoding="utf-8").write(page)
print("Tableau de bord prêt : index.html")
