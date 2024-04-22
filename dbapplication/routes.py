from flask import request, jsonify
from models import Vehicle, TimeLog
from datetime import datetime, timedelta

def register_routes(app, db):

  @app.route('/vehicle', methods=['GET', 'POST'])
  def index():
    if request.method == 'GET':
      vehicles = Vehicle.query.all()
      vehicle_list = [vehicle.to_dict() for vehicle in vehicles]
      return jsonify(vehicle_list)
    elif request.method == 'POST':
      name = request.json['name']
      plate_number = request.json['plate_number']
      car_brand = request.json['car_brand']
      new_vehicle = Vehicle(name=name.lower(), plate_number=plate_number.lower().strip(), car_brand=car_brand.lower())
      db.session.add(new_vehicle)
      db.session.commit()
      return str(new_vehicle)
    
  # # Delete a vehicle
  @app.route('/vehicle/<id>', methods=['DELETE'])
  def delete_vehicle(id):
      vehicle = Vehicle.query.get(id)
      db.session.delete(vehicle)
      db.session.commit()
      return str(vehicle)
  
  @app.route('/vehicle/plate-number/log', methods=['POST'])
  def log_vehicle():
    plate_number = request.json['plate_number']
    vehicle = Vehicle.query.filter_by(plate_number=plate_number).first()
    if vehicle:
        # Check if there is a previous time log for this vehicle
        previous_log = TimeLog.query.filter_by(vehicle_id=vehicle.id).order_by(TimeLog.log_time.desc()).first()
        if not previous_log or (datetime.utcnow() - previous_log.log_time) > timedelta(seconds=15):
            stat = 'IN'
            if previous_log:
              stat = 'IN' if previous_log.status == 'OUT' else 'OUT'
            
            time_log = TimeLog(vehicle_id=vehicle.id, status=stat)
            db.session.add(time_log)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Vehicle logged successfully.'}), 200
    return jsonify({'success': False, 'message': 'Failed to log vehicle.'}), 400
  
  @app.route('/vehicle/logs', methods=['GET'])
  def index_logs():
    if request.method == 'GET':
      logs = TimeLog.query.order_by(TimeLog.log_time.desc()).all()
      logs_list = [log.to_dict() for log in logs]
      return jsonify(logs_list)

# def add_vehicle():
#     name = request.json['name']
#     plate_number = request.json['plate_number']
#     car_brand = request.json['car_brand']
#     new_vehicle = Vehicle(name, plate_number, car_brand)
#     db.session.add(new_vehicle)
#     db.session.commit()
#     return vehicle_schema.jsonify(new_vehicle)

# # Get all vehicles
# @app.route('/vehicle', methods=['GET'])
# def get_vehicles():
#     all_vehicles = Vehicle.query.all()
#     result = vehicles_schema.dump(all_vehicles)
#     return jsonify(result)

# # Get single vehicle
# @app.route('/vehicle/<id>', methods=['GET'])
# def get_vehicle(id):
#     vehicle = Vehicle.query.get(id)
#     return vehicle_schema.jsonify(vehicle)

# # Update a vehicle
# @app.route('/vehicle/<id>', methods=['PUT'])
# def update_vehicle(id):
#     vehicle = Vehicle.query.get(id)
#     name = request.json['name']
#     plate_number = request.json['plate_number']
#     car_brand = request.json['car_brand']
#     vehicle.name = name
#     vehicle.plate_number = plate_number
#     vehicle.car_brand = car_brand
#     db.session.commit()
#     return vehicle_schema.jsonify(vehicle)

# # Delete a vehicle
# @app.route('/vehicle/<id>', methods=['DELETE'])
# def delete_vehicle(id):
#     vehicle = Vehicle.query.get(id)
#     db.session.delete(vehicle)
#     db.session.commit()
#     return vehicle_schema.jsonify(vehicle)

