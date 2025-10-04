# spacemonkeynasa

# 1. Clone the repo
git clone https://github.com/username/my-django-app.git
cd backend

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# OR
.venv\Scripts\activate      # Windows PowerShell

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the development server
python manage.py runserver
