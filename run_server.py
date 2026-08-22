import os
import sys
import uvicorn
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

if __name__ == "__main__":
    print("[SERVER] Starting ApexRAG Multimodal Server on http://localhost:8000 ...")
    uvicorn.run(
        "apexrag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
