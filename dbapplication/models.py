from app import db

class Vehicle(db.Model):
  __tablename__ = 'vehicles'

  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.Text, nullable=False)
  plate_number = db.Column(db.Text, nullable=True)
  car_brand = db.Column(db.Text, nullable=True)

  def __repr__(self) -> str:
    return f'Vehicle owned by {self.name} with a plate of {self.plate_number}'
  
  def to_dict(self):
    return {
      'id': self.id,
      'name': self.name,
      'plate_number': self.plate_number,
      'car_brand': self.car_brand
    }