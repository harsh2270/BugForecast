import re
for f in ("ms_report.html", "flask_report.html", "requests_report.html"):
    html = open(f, encoding="utf-8").read()
    m = re.search(r"class='v'>([0-9.]+)</div><div class='l'>ensemble AUC", html)
    commits = re.search(r"class='v'>([0-9]+)</div><div class='l'>commits analyzed", html)
    print(f, "| AUC:", m.group(1) if m else "?", "| commits:", commits.group(1) if commits else "?")
