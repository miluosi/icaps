from typing import Optional, List, Dict
import time


class ChargingStation:
    def __init__(self, id: int, location: int, max_capacity: int = 10):
        self.id = id
        self.location = location
        self.max_capacity = max_capacity
        self.current_vehicles: List[str] = []
        self.charging_queue: List[str] = []
        self.charging_queue_notarrived: List[str] = []
        self.available_slots = max_capacity

    def is_available(self) -> bool:
        """Check if there are available charging slots"""
        return self.available_slots > 0
    
    def add_to_queue(self, vehicle_id: str) -> bool:
        """Add vehicle to charging queue"""
        if vehicle_id not in self.charging_queue and vehicle_id not in self.current_vehicles:
            self.charging_queue.append(vehicle_id)
            return True
        return False

    def start_charging(self, vehicle_id: str) -> bool:
        """Start charging a vehicle if slots are available"""
        if self.is_available():
            if vehicle_id in self.charging_queue:
                self.charging_queue.remove(vehicle_id)
            
            if vehicle_id not in self.current_vehicles:
                self.current_vehicles.append(vehicle_id)
                self.available_slots -= 1
                # print(f"Vehicle {vehicle_id} started charging at station {self.id}")
                return True
        else:
            # Add to queue if not available
            if self.add_to_queue(vehicle_id):
                print(f"Station {self.id} is full. Vehicle {vehicle_id} added to queue.")
        return False

    def stop_charging(self, vehicle_id: str) -> bool:
        """Stop charging a vehicle and free up slot"""
        if vehicle_id in self.current_vehicles:
            self.current_vehicles.remove(vehicle_id)
            self.available_slots += 1
            # print(f"Vehicle {vehicle_id} finished charging at station {self.id}")
            
            # Start charging next vehicle in queue if available
            if self.charging_queue and self.is_available():
                next_vehicle = self.charging_queue.pop(0)
                self.start_charging(next_vehicle)
            
            return True
        return False

    def get_station_status(self) -> Dict:
        """Get current station status"""
        return {
            'station_id': self.id,
            'location': self.location,
            'available_slots': self.available_slots,
            'max_capacity': self.max_capacity,
            'current_vehicles': self.current_vehicles.copy(),
            'queue_length': len(self.charging_queue),
            'utilization_rate': (self.max_capacity - self.available_slots) / self.max_capacity
        }

    def estimate_wait_time(self) -> float:
        """Estimate wait time for new vehicles (in minutes)"""
        if self.is_available():
            return 0.0
        
        # Simple estimation: average charging time * queue position
        avg_charging_time = 30.0  # minutes
        return len(self.charging_queue) * avg_charging_time


class ChargingStationManager:
    """Manages multiple charging stations"""
    
    def __init__(self):
        self.stations: Dict[int, ChargingStation] = {}
        
    def add_station(self, station_id: int, location: int, capacity: int = 4) -> None:
        """Add a new charging station"""
        self.stations[station_id] = ChargingStation(station_id, location, capacity)
        
    def get_nearest_available_station(self, vehicle_location: int) -> Optional[ChargingStation]:
        """Find the nearest available charging station"""
        available_stations = [station for station in self.stations.values() 
                            if station.is_available()]
        
        if not available_stations:
            return None
            
        # Simple distance calculation (can be replaced with actual distance function)
        nearest_station = min(available_stations, 
                            key=lambda s: abs(s.location - vehicle_location))
        return nearest_station
    
    def get_station_with_shortest_queue(self) -> Optional[ChargingStation]:
        """Find station with shortest queue"""
        if not self.stations:
            return None
            
        return min(self.stations.values(), 
                  key=lambda s: len(s.charging_queue) + (s.max_capacity - s.available_slots))
    
    def get_all_stations_status(self) -> List[Dict]:
        """Get status of all stations"""
        return [station.get_station_status() for station in self.stations.values()]
    
    def update_all_stations(self) -> None:
        """Update all charging stations (for environment simulation)"""
        # 简单的更新逻辑 - 可以扩展为更复杂的状态更新
        for station in self.stations.values():
            # 这里可以添加时间相关的更新逻辑
            # 例如：队列处理、充电进度更新等
            pass
