import sys
import os
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
