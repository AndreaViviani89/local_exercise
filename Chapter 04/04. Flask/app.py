from flask import Flask, render_template

# create an instance
app = Flask(__name__)

@app.route("/") ## Decorator 
def index():
    return render_template('home.html')


# @app.route("/about") ## Decorator 
@app.route("/about") ## Decorator 
def about():
    return render_template('about.html')


@app.route("/portfolio")
def portolio():
    return render_template('portfolio.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)