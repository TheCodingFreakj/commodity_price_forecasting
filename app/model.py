# app/model.py
def create_commodity_price_model(db):
    class CommodityPrice(db.Model):
        __tablename__ = 'commodity_price'

        id = db.Column(db.Integer, primary_key=True)
        symbol = db.Column(db.String(10), nullable=False)  # Adjust length if needed
        date = db.Column(db.DateTime, nullable=False)
        open_price = db.Column(db.Float)
        high_price = db.Column(db.Float)
        low_price = db.Column(db.Float)
        close_price = db.Column(db.Float)
        volume = db.Column(db.Integer)

        def __repr__(self):
            return f'<CommodityPrice {self.symbol} {self.date} {self.close_price}>'

    return CommodityPrice
