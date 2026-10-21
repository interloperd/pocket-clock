"""Intègre dashboard_data.json dans le template et produit pocket_clock_dashboard.html."""
html = open("dashboard_template.html", encoding="utf-8").read()
data = open("dashboard_data.json", encoding="utf-8").read().replace("NaN", "null")
out = html.replace("/*DATA*/null", data)
for f in ("pocket_clock_dashboard.html", "index.html"):
    open(f, "w", encoding="utf-8").write(out)
print("Tableau de bord prêt : pocket_clock_dashboard.html")
