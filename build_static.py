"""把兩個 API 的回應存成靜態 JSON，供 GitHub Pages 使用。改了 web_demo.py 後要重跑。"""
from pathlib import Path
import web_demo

c = web_demo.app.test_client()
for api, name in [("analyze-document", "analyze-document"), ("compare-methods", "compare-methods")]:
    r = c.get(f"/api/{api}")
    assert r.status_code == 200
    Path("static-data", f"{name}.json").write_bytes(r.data)
    print(name, len(r.data) // 1024, "KB")
