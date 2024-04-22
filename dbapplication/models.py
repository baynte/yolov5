from app import db
from datetime import datetime

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

class TimeLog(db.Model):
    __tablename__ = 'time_logs'

    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicles.id'), nullable=False)
    log_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    status = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'status': self.status,
            'log_time': self.log_time.strftime('%Y-%m-%d %H:%M:%S')
        }