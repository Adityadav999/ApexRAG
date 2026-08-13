import sys
import os
import site

# Dynamically ensure user site-packages directory is present in sys.path
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

__version__ = "1.0.0"
