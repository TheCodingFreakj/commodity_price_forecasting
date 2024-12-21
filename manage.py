from flask.cli import FlaskGroup
from app import create_app, db

app = create_app()

cli = FlaskGroup(create_app=create_app)

@cli.command("db")
def db_command():
    db.create_all()

if __name__ == "__main__":
    cli()
