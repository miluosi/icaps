"""Lossless numeric histories; materialize row dictionaries only when read."""
from collections.abc import Sequence
import struct


class _PackedRows(Sequence):
    def __init__(self, rows=()):
        self._data = bytearray()
        self.extend(rows)

    def __len__(self):
        return len(self._data) // self._format.size

    def append(self, row):
        self._data.extend(self._format.pack(*self._encode(row)))

    def extend(self, rows):
        if type(rows) is type(self):
            self._data.extend(rows._data)
        else:
            for row in rows:
                self.append(row)

    def __iter__(self):
        for values in self._format.iter_unpack(self._data):
            yield self._decode(values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._decode(self._format.unpack_from(self._data, index * self._format.size))


class PositionHistory(_PackedRows):
    _format = struct.Struct('<qddd')

    @staticmethod
    def _encode(row):
        return int(row['zone']), float(row['time']), *row['coordinates']

    @staticmethod
    def _decode(values):
        zone, time, lat, lon = values
        return dict(zone=zone, coordinates=(lat, lon), time=time)


class ChargingEventHistory(_PackedRows):
    _format = struct.Struct('<qqdd')

    @staticmethod
    def _encode(row):
        return int(row['vehicle_id']), int(row['station_id']), float(row['duration']), float(row['time'])

    @staticmethod
    def _decode(values):
        vehicle, station, duration, time = values
        return dict(vehicle_id=vehicle, station_id=station, duration=duration, time=time)
