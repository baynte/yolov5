from flask import request, jsonify
from models import Vehicle

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

