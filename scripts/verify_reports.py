import re
for f in ("ms_report.html", "flask_report.html", "requests_report.html"):
    html = open(f, encoding="utf-8").read()
    tabs = len(re.findall(r"<button onclick=\"tab\(", html))
    panes = html.count("class='tabpane")
    eff = re.search(r"<div class='v'>([0-9]+%)</div><div class='l'>defects caught", html)
    drivers = html.count("Top risk drivers")
    print(f, "| tabs:", tabs, "| panes:", panes, "| eff20:",
          eff.group(1) if eff else "?", "| driver-col:", bool(drivers))
