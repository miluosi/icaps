class Request(object):
    """
    A Request is the atomic unit in an Action.

    It represents a single customer's *request* for a ride
    """

    MAX_PICKUP_DELAY: float =2
    MAX_DROPOFF_DELAY: float = 150.0  # Increased from 60 to 150 - allow time to complete trip

    def __init__(self,
                 request_id: int,
                 source: int,
                 destination: int,
                 current_time: float,
                 travel_time: float,
                 value: float=10,
                 final_value: float=10,
                 trip_distance_km: float | None = None,
                 ):
        self.request_id = request_id
        self.pickup = source
        self.dropoff = destination
        self.value = value  # In the deafult case, all requests have equal value
        self.pickup_deadline = current_time + self.MAX_PICKUP_DELAY
        self.dropoff_deadline = current_time + travel_time + self.MAX_DROPOFF_DELAY
        self.travel_time = float(travel_time)
        self.final_value = final_value  # 修正：应该使用final_value参数而不是value
        self.trip_distance_km = (
            None if trip_distance_km is None else max(0.0, float(trip_distance_km))
        )
        self.created_time = current_time
        self.assigned_time = None
        self.pickup_time = None
        self.completed_time = None
        self.ev_rejection_count = 0
        self.was_ev_rejected = False
        self.completed_by_vehicle_type = None
    def __deepcopy__(self, memo):
        return self

    def __str__(self):
        return("{}->{}".format(self.pickup, self.dropoff))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.request_id)

    def __eq__(self, other):
        # Request is only comparable with other Requests
        if isinstance(other, self.__class__):
            # If the ids are the same, they are equal
            if (self.request_id == other.request_id):
                return True
        
        return False
