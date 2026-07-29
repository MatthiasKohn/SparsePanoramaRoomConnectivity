"""Connectivity provider — which panos belong to which room, and room adjacency (shared doors).

get_rooms(fl, panos, model) -> (rooms: {room_id: [pano_stem,...]}, adj: {room_id: set(room_id)})

Models:
  gt       : ZInD room membership + adjacency from GT shared doors (the oracle).
  detected : from detected doors + our connectivity graph (not yet wired -> raises).

`adj` powers the room-aware renderer (show current room + door-adjacent rooms -> no see-through).
"""
import numpy as np


def _adjacency(fl, rooms):
    """Two rooms are adjacent if a pano of one shares a door with a pano of the other."""
    room_of = {s: r for r, ss in rooms.items() for s in ss}
    adj = {r: set() for r in rooms}
    stems = list(room_of)
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            a, b = stems[i], stems[j]
            if room_of[a] == room_of[b]:
                continue
            if fl.shared_door(a, b, tol=0.25) is not None:
                adj[room_of[a]].add(room_of[b]); adj[room_of[b]].add(room_of[a])
    return adj


def get_rooms(fl, panos, model="gt"):
    if model != "gt":
        raise NotImplementedError("connectivity=detected: run door detection + the connectivity "
                                  "graph and map panos->rooms; only 'gt' is wired for now.")
    rooms = {}
    for s in panos:
        rooms.setdefault(fl.panos[s]["room"], []).append(s)
    return rooms, _adjacency(fl, rooms)
