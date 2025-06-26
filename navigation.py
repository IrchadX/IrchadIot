
import json
import math
import heapq
import time
import threading
import paho.mqtt.client as mqtt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import numpy as np
from picamera2 import Picamera2
import math
import threading

from obstacle_detection import ObstacleDetectionSystem, DetectedObstacle
from position_tracking import EnhancedContinuousIMUTracker, ContinuousPosition
import numpy as np

# Constants
MQTT_BROKER_URL = "mqtts://56e91a6cae9041d89eca972b135301fb.s1.eu.hivemq.cloud:8883"
MQTT_USERNAME = "hind_deh"
MQTT_PASSWORD = "hindDehili2025"
MQTT_TOPIC_REQUEST = "navigation/request"
MQTT_TOPIC_INSTRUCTIONS = "navigation/instructions"
MQTT_TOPIC_OBSTACLE = "navigation/obstacle"

# Navigation parameters
PATH_UPDATE_INTERVAL = 1.0  # seconds
OBSTACLE_AVOIDANCE_RADIUS = 0.5  # meters
POSITION_TOLERANCE = 0.3  # meters for re-routing
TURN_THRESHOLD_DEG = 15.0  # degrees

@dataclass
class Waypoint:
    x: float
    y: float
    is_poi: bool = False
    poi_id: Optional[int] = None
    poi_name: Optional[str] = None

# Add these new constants
REAL_TIME_INSTRUCTION_INTERVAL = 0.5  # Send instructions every 0.5 seconds
ENVIRONMENT_UPDATE_INTERVAL = 2.0     # Update environment info every 2 seconds

@dataclass
class NavigationInstruction:
    command: str  # "turn", "move", "arrived"
    direction: Optional[str] = None  # "left", "right", "straight"
    distance: Optional[float] = None  # meters
    angle: Optional[float] = None  # degrees
    next_waypoint: Optional[Waypoint] = None
    instruction_text: Optional[str] = None  # Add this line

class SharedCamera:
    def __init__(self, resolution=(640, 480)):
        self.picam2 = None
        self.resolution = resolution
        self.ref_count = 0
        self.lock = threading.Lock()
    
    def initialize(self):
        with self.lock:
            if self.picam2 is None:
                try:
                    self.picam2 = Picamera2()
                    config = self.picam2.create_preview_configuration(
                        main={"size": self.resolution, "format": "RGB888"},
                        controls={"FrameRate": 30}
                    )
                    self.picam2.configure(config)
                    self.picam2.start()
                    print(f"Shared camera initialized at {self.resolution} resolution")
                    self.ref_count = 1
                    return True
                except Exception as e:
                    print(f"Camera initialization failed: {e}")
                    self.picam2 = None
                    return False
            else:
                self.ref_count += 1
                return True
    
    def release(self):
        with self.lock:
            if self.picam2 is not None:
                self.ref_count -= 1
                if self.ref_count <= 0:
                    self.picam2.stop()
                    self.picam2 = None
                    print("Shared camera released")
    
    def capture_array(self):
        with self.lock:
            if self.picam2 is not None:
                return self.picam2.capture_array()
        return None

class GeoJSONParser:
    def __init__(self, geojson_data: Dict):
        self.geojson = geojson_data
        self.walkable_zones = []
        self.walls = []
        self.pois = []
        self.navigation_graph = {}
        self.local_reference = None
        
        # Extract features
        self._parse_features()
        
        # Convert coordinates to local 2D system
        self._convert_to_local_coordinates()
        
        # Build navigation graph
        self._build_navigation_graph()
    
    def _parse_features(self):
        """Parse GeoJSON features into categories"""
        for feature in self.geojson.get('features', []):
            prop = feature.get('properties', {})
            geom = feature.get('geometry', {})
            
            if prop.get('type') == 'zone' and prop.get('accessible', False):
                self.walkable_zones.append((geom['coordinates'], prop))
            elif prop.get('type') == 'wall':
                self.walls.append((geom['coordinates'], prop))
            elif prop.get('type') == 'poi':
                self.pois.append((geom['coordinates'], prop))
    
    def _convert_to_local_coordinates(self):
        """Convert GPS coordinates to local 2D Cartesian system"""
        if not self.walkable_zones:
            return
            
        # Use first point of first walkable zone as reference
        ref_point = self.walkable_zones[0][0][0][0]
        self.local_reference = ref_point
        
        # Convert all coordinates
        def convert_point(point):
            # Simple conversion - for small indoor areas, we can approximate
            # Earth's curvature as flat and use meters for delta lat/lon
            lon, lat = point
            delta_lon = (lon - ref_point[0]) * 111320 * math.cos(math.radians(lat))
            delta_lat = (lat - ref_point[1]) * 111000  # Approx meters per degree latitude
            return (delta_lon, delta_lat)
        
        # Convert walkable zones
        converted_zones = []
        for zone, props in self.walkable_zones:
            converted_zone = []
            for polygon in zone:  # GeoJSON Polygon has outer ring + holes
                converted_polygon = [convert_point(point) for point in polygon]
                converted_zone.append(converted_polygon)
            converted_zones.append((converted_zone, props))
        self.walkable_zones = converted_zones
        
        # Convert walls
        converted_walls = []
        for wall, props in self.walls:
            if wall['type'] == 'LineString':
                converted_wall = [convert_point(point) for point in wall['coordinates']]
            else:  # Polygon
                converted_wall = []
                for polygon in wall['coordinates']:
                    converted_wall.append([convert_point(point) for point in polygon])
            converted_walls.append((converted_wall, props))
        self.walls = converted_walls
        
        # Convert POIs - FIX: Ensure we return tuples, not lists
        converted_pois = []
        for poi, props in self.pois:
            if poi['type'] == 'Point':
                converted_poi = convert_point(poi['coordinates'])
            else:  # Polygon or LineString
                if poi['type'] == 'Polygon':
                    # Use centroid of first polygon
                    points = poi['coordinates'][0]
                else:  # LineString
                    points = poi['coordinates']
                centroid_lon = sum(p[0] for p in points) / len(points)
                centroid_lat = sum(p[1] for p in points) / len(points)
                converted_poi = convert_point((centroid_lon, centroid_lat))
            
            # Ensure converted_poi is a tuple
            converted_pois.append((tuple(converted_poi), props))
        self.pois = converted_pois
    
    def _build_navigation_graph(self):
        """Build a navigation graph from walkable zones"""
        if not self.walkable_zones:
            return
        
        # Create a grid of points within walkable zones
        grid_spacing = 0.5  # meters
        grid_points = []
        
        for zone, _ in self.walkable_zones:
            # Get bounding box of zone
            all_points = [point for polygon in zone for point in polygon]
            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)
            
            # Generate grid points
            x = min_x
            while x <= max_x:
                y = min_y
                while y <= max_y:
                    if self._point_in_polygon((x, y), zone[0]):  # Only check outer polygon
                        grid_points.append((float(x), float(y)))  # Ensure floats
                    y += grid_spacing
                x += grid_spacing
        
        # Add POIs as additional nodes (ensure they are tuples of floats)
        for poi, props in self.pois:
            if isinstance(poi, (tuple, list)):
                grid_points.append((float(poi[0]), float(poi[1])))
            else:
                # Handle numpy arrays or other types
                grid_points.append((float(poi[0]), float(poi[1])))
        
        print(f"Generated {len(grid_points)} navigation points")
        
        # Create graph edges between nearby points
        self.navigation_graph = {}
        connection_distance = grid_spacing * 1.5
        
        # Initialize graph with empty neighbor lists
        for point in grid_points:
            clean_point = (float(point[0]), float(point[1]))
            self.navigation_graph[clean_point] = []
        
        # Create connections
        for i, point1 in enumerate(grid_points):
            point1 = (float(point1[0]), float(point1[1]))
            for point2 in grid_points[i+1:]:
                point2 = (float(point2[0]), float(point2[1]))
                distance = math.hypot(point1[0]-point2[0], point1[1]-point2[1])
                if distance <= connection_distance and not self._line_intersects_walls(point1, point2):
                    self.navigation_graph[point1].append((point2, float(distance)))
                    self.navigation_graph[point2].append((point1, float(distance)))
        
        print(f"Navigation graph built with {len(self.navigation_graph)} nodes")
        
    def _point_in_polygon(self, point, polygon) -> bool:
        """Check if a point is inside a polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n+1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _line_intersects_walls(self, point1, point2) -> bool:
        """Check if a line between two points intersects any walls"""
        for wall, _ in self.walls:
            if isinstance(wall[0], list):  # Polygon wall
                wall_segments = []
                polygon = wall[0]  # Only check outer polygon
                for i in range(len(polygon)):
                    wall_segments.append((polygon[i-1], polygon[i]))
            else:  # LineString wall
                wall_segments = []
                for i in range(1, len(wall)):
                    wall_segments.append((wall[i-1], wall[i]))
            
            for seg_start, seg_end in wall_segments:
                if self._line_segments_intersect(point1, point2, seg_start, seg_end):
                    return True
        return False
    
    @staticmethod
    def _line_segments_intersect(a1, a2, b1, b2) -> bool:
        """Check if two line segments intersect"""
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        
        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)
    
    def find_closest_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Find the closest point in the navigation graph to the given point"""
        if not self.navigation_graph:
            return point
            
        min_dist = float('inf')
        closest_point = None
        
        # Ensure input point is a tuple
        if isinstance(point, (list, np.ndarray)):
            point = tuple(point)
        
        for graph_point in self.navigation_graph.keys():
            # Ensure graph_point is also a tuple
            if isinstance(graph_point, (list, np.ndarray)):
                graph_point = tuple(graph_point)
                
            dist = math.hypot(point[0]-graph_point[0], point[1]-graph_point[1])
            if dist < min_dist:
                min_dist = dist
                closest_point = graph_point
        
        return closest_point
    
    def find_poi_by_id(self, poi_id: int) -> Optional[Tuple[Tuple[float, float], Dict]]:
        """Find a POI by its ID"""
        for poi, props in self.pois:
            if props.get('id') == poi_id:
                # Ensure poi is a tuple
                if isinstance(poi, (list, np.ndarray)):
                    poi = tuple(poi)
                return (poi, props)
        return None

class IndoorNavigationSystem:
    def __init__(self, geojson_data: Dict):
        self.map = GeoJSONParser(geojson_data)
         # Create shared camera first
        self.shared_camera = SharedCamera(resolution=(640, 480))
        
        # Pass shared camera to both systems
        self.position_tracker = EnhancedContinuousIMUTracker(shared_camera=self.shared_camera)
        self.obstacle_detector = ObstacleDetectionSystem(shared_camera=self.shared_camera)
        
        self.mqtt_client = None
        
        # Navigation state
        self.current_path = []
        self.current_waypoint_index = 0
        self.destination = None
        self.last_path_update = 0
        self.active_obstacles = []
        
        # Threading
        self.running = False
        self.navigation_thread = None
    
    def initialize(self) -> bool:
        """Initialize all components"""
        # Initialize shared camera first
        if not self.shared_camera.initialize():
            print("Failed to initialize shared camera")
            return False
            
        # Then initialize other components
        if not self.position_tracker.initialize():
            print("Failed to initialize position tracker")
            return False
            
        if not self.obstacle_detector.initialize():
            print("Failed to initialize obstacle detector")
            return False
            
        # Setup MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.mqtt_client.tls_set()  # Enable TLS
        
        try:
            self.mqtt_client.connect(
                host="56e91a6cae9041d89eca972b135301fb.s1.eu.hivemq.cloud",
                port=8883
            )
            self.mqtt_client.subscribe(MQTT_TOPIC_REQUEST)
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            return False
        
        # Start position tracking and obstacle detection
        self.position_tracker.start_tracking()
        self.obstacle_detector.start_detection(
            current_heading_callback=lambda: self.position_tracker.get_current_position().yaw
        )
        
        # Register obstacle callback
        self.obstacle_detector.add_path_blocking_callback(self._on_obstacle_detected)
        
        return True
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            payload = json.loads(msg.payload.decode())
            if msg.topic == MQTT_TOPIC_REQUEST:
                self._handle_navigation_request(payload)
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def _handle_navigation_request(self, request: Dict):
        """Handle navigation request from mobile app"""
        destination_poi_id = request.get('destination_id')
        if destination_poi_id is None:
            print("No destination POI ID provided")
            return
            
        # Find the destination POI
        poi = self.map.find_poi_by_id(destination_poi_id)
        if not poi:
            print(f"POI with ID {destination_poi_id} not found")
            return
            
        self.destination = Waypoint(
            x=poi[0][0],
            y=poi[0][1],
            is_poi=True,
            poi_id=destination_poi_id,
            poi_name=poi[1].get('name', 'Destination')
        )
        
        print(f"Navigation requested to: {self.destination.poi_name}")
        
        # Plan initial path
        self._plan_path_to_destination()
        
        # Start navigation if not already running
        if not self.running:
            self.start_navigation()
    
    def _on_obstacle_detected(self, obstacles: List[DetectedObstacle]):
        """Handle newly detected obstacles that block the path"""
        current_pos = self.position_tracker.get_current_position()
        current_heading = current_pos.yaw
        
        # Convert obstacle positions to global coordinates
        global_obstacles = []
        for obs in obstacles:
            # Convert relative position to global coordinates
            angle_rad = math.radians(current_heading + obs.angle)
            global_x = current_pos.x + obs.distance * math.sin(angle_rad)
            global_y = current_pos.y + obs.distance * math.cos(angle_rad)
            
            global_obstacles.append((global_x, global_y, obs.size[0], obs.size[1]))
            
            # Send obstacle alert via MQTT
            self._send_obstacle_alert(obs)
        
        # Add to active obstacles
        self.active_obstacles.extend(global_obstacles)
        
        # Check if any obstacle is on our current path
        if self._is_path_blocked():
            print("Obstacle detected on path - replanning...")
            self._plan_path_to_destination()
    
    def _send_obstacle_alert(self, obstacle: DetectedObstacle):
        """Send obstacle alert to mobile app via MQTT"""
        alert = {
            'type': obstacle.type,
            'object_class': obstacle.object_class,
            'distance': obstacle.distance,
            'angle': obstacle.angle,
            'timestamp': obstacle.timestamp
        }
        
        try:
            self.mqtt_client.publish(
                MQTT_TOPIC_OBSTACLE,
                json.dumps(alert))
        except Exception as e:
            print(f"Failed to send obstacle alert: {e}")
    
    def _is_path_blocked(self) -> bool:
        """Check if current path is blocked by any obstacle"""
        if not self.current_path or not self.active_obstacles:
            return False
            
        current_pos = self.position_tracker.get_current_position()
        current_point = (current_pos.x, current_pos.y)
        
        # Check path segments starting from current waypoint
        for i in range(self.current_waypoint_index, len(self.current_path)-1):
            segment_start = (self.current_path[i].x, self.current_path[i].y)
            segment_end = (self.current_path[i+1].x, self.current_path[i+1].y)
            
            for obs_x, obs_y, obs_w, obs_h in self.active_obstacles:
                # Simple check - if obstacle is within avoidance radius of path segment
                distance = self._distance_point_to_line_segment(
                    (obs_x, obs_y), segment_start, segment_end)
                
                if distance < OBSTACLE_AVOIDANCE_RADIUS:
                    return True
        
        return False
    
    @staticmethod
    def _distance_point_to_line_segment(point, line_start, line_end) -> float:
        """Calculate distance from point to line segment"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line length squared
        l2 = (x2-x1)**2 + (y2-y1)**2
        if l2 == 0:
            return math.hypot(px-x1, py-y1)
        
        # Consider the line extending the segment, parameterized as start + t*(end-start)
        t = max(0, min(1, ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / l2))
        
        # Projection point on the line
        proj_x = x1 + t*(x2-x1)
        proj_y = y1 + t*(y2-y1)
        
        return math.hypot(px-proj_x, py-proj_y)
    
    def _plan_path_to_destination(self):
        """Plan path from current position to destination using A* algorithm"""
        if not self.destination:
            return
            
        current_pos = self.position_tracker.get_current_position()
        start_point = (float(current_pos.x), float(current_pos.y))
        
        # Find closest points in navigation graph
        graph_start = self.map.find_closest_point(start_point)
        graph_end = self.map.find_closest_point((float(self.destination.x), float(self.destination.y)))
        
        if not graph_start or not graph_end:
            print("Failed to find path - start or end point not in walkable area")
            return
        
        # Ensure points are tuples of floats
        def ensure_tuple(point):
            if isinstance(point, (list, np.ndarray)):
                return tuple(float(x) for x in point)
            elif isinstance(point, tuple):
                return tuple(float(x) for x in point)
            return point
        
        graph_start = ensure_tuple(graph_start)
        graph_end = ensure_tuple(graph_end)
            
        print(f"Planning path from {graph_start} to {graph_end}")
        print(f"Start type: {type(graph_start)}, End type: {type(graph_end)}")
            
        # Run A* algorithm
        path = self._a_star_pathfinding(graph_start, graph_end)
        
        if not path:
            print("No valid path found to destination")
            return
            
        # Convert path to waypoints
        self.current_path = []
        for point in path:
            # Ensure each point is a tuple of floats
            point = ensure_tuple(point)
            self.current_path.append(Waypoint(x=float(point[0]), y=float(point[1])))
        
        # Add destination as final waypoint
        self.current_path.append(self.destination)
        self.current_waypoint_index = 0
        self.last_path_update = time.time()
        
        # Print path information
        print(f"\n[PATH] New path planned with {len(self.current_path)} waypoints:")
        for i, wp in enumerate(self.current_path):
            poi_info = f" ({wp.poi_name})" if wp.is_poi else ""
            print(f"  {i+1}. ({wp.x:.2f}, {wp.y:.2f}){poi_info}")

        
    def _a_star_pathfinding(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """A* pathfinding algorithm"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {point: float('inf') for point in self.map.navigation_graph.keys()}
        g_score[start] = 0
        
        f_score = {point: float('inf') for point in self.map.navigation_graph.keys()}
        f_score[start] = self._heuristic(start, goal)
        
        open_set_hash = {start}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
                
            for neighbor, distance in self.map.navigation_graph[current]:
                # Skip if neighbor is too close to an obstacle
                if self._is_point_near_obstacle(neighbor):
                    continue
                    
                tentative_g_score = g_score[current] + distance
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return []  # No path found
    
    def _is_point_near_obstacle(self, point: Tuple[float, float]) -> bool:
        """Check if a point is too close to any active obstacle"""
        for obs_x, obs_y, obs_w, obs_h in self.active_obstacles:
            distance = math.hypot(point[0]-obs_x, point[1]-obs_y)
            if distance < OBSTACLE_AVOIDANCE_RADIUS:
                return True
        return False
    
    @staticmethod
    def _heuristic(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance heuristic for A* that handles NumPy arrays and nested lists"""
        # Handle different input types
        try:
            # If a or b are nested lists/tuples, extract the first element
            if isinstance(a, (list, tuple)) and len(a) > 0:
                if isinstance(a[0], (list, tuple)):
                    a = a[0]  # Take first element if nested
            if isinstance(b, (list, tuple)) and len(b) > 0:
                if isinstance(b[0], (list, tuple)):
                    b = b[0]  # Take first element if nested
            
            # Convert to plain Python floats
            ax = float(a[0])
            ay = float(a[1])
            bx = float(b[0])
            by = float(b[1])
            
            return math.hypot(ax - bx, ay - by)
        except (TypeError, IndexError, ValueError) as e:
            print(f"Error in heuristic calculation: {e}")
            print(f"a = {a}, type = {type(a)}")
            print(f"b = {b}, type = {type(b)}")
            return float('inf')

    
    def start_navigation(self):
        """Start the navigation system"""
        if self.running:
            return
            
        self.running = True
        self.navigation_thread = threading.Thread(target=self._navigation_loop, daemon=True)
        self.navigation_thread.start()
        print("Navigation system started")
    
    def stop_navigation(self):
        """Stop the navigation system"""
        self.running = False
        if self.navigation_thread:
            self.navigation_thread.join(timeout=1.0)
        
        self.position_tracker.stop_tracking()
        self.obstacle_detector.stop_detection()
        
        # Release shared camera
        self.shared_camera.release()
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        print("Navigation system stopped")
    
    def _navigation_loop(self):
        """Main navigation loop"""
        while self.running:
            try:
                # Get current position
                current_pos = self.position_tracker.get_current_position()
                
                # Check if we need to replan path
                if (time.time() - self.last_path_update > PATH_UPDATE_INTERVAL or 
                    self._has_position_diverged()):
                    self._plan_path_to_destination()
                
                # If we have a path, navigate
                if self.current_path and self.current_waypoint_index < len(self.current_path):
                    self._navigate_to_next_waypoint(current_pos)
                
                # Clean up old obstacles
                self._cleanup_old_obstacles()
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Navigation loop error: {e}")
                time.sleep(1.0)
    
    def _has_position_diverged(self) -> bool:
        """Check if current position has diverged from path beyond tolerance"""
        if not self.current_path or self.current_waypoint_index >= len(self.current_path):
            return False
            
        current_pos = self.position_tracker.get_current_position()
        current_point = (current_pos.x, current_pos.y)
        
        # Find closest point on path
        min_dist = float('inf')
        closest_segment = None
        
        for i in range(max(0, self.current_waypoint_index-1), len(self.current_path)-1):
            segment_start = (self.current_path[i].x, self.current_path[i].y)
            segment_end = (self.current_path[i+1].x, self.current_path[i+1].y)
            
            distance = self._distance_point_to_line_segment(current_point, segment_start, segment_end)
            if distance < min_dist:
                min_dist = distance
                closest_segment = i
        
        return min_dist > POSITION_TOLERANCE
    
    def _navigate_to_next_waypoint(self, current_pos: ContinuousPosition):
        """Navigate to the next waypoint in the path"""
        current_point = (current_pos.x, current_pos.y)
        next_waypoint = self.current_path[self.current_waypoint_index]
        next_point = (next_waypoint.x, next_waypoint.y)
        
        # Calculate distance to waypoint
        distance = math.hypot(next_point[0]-current_point[0], next_point[1]-current_point[1])
        
        if distance < POSITION_TOLERANCE:
            # Reached waypoint
            self.current_waypoint_index += 1
            
            if self.current_waypoint_index >= len(self.current_path):
                # Reached destination
                instruction = NavigationInstruction(
                    command="arrived",
                    next_waypoint=self.destination,
                    instruction_text=f"You have arrived at {self.destination.poi_name}"
                )
                self._send_navigation_instruction(instruction)
                print(f"\n[ARRIVAL] {instruction.instruction_text}\n")
                print("Destination reached!")
                self.current_path = []
                self.current_waypoint_index = 0
                self.destination = None
            else:
                # Moving to next waypoint
                next_waypoint = self.current_path[self.current_waypoint_index]
                instruction = self._calculate_instruction(current_pos, next_waypoint)
                self._send_navigation_instruction(instruction)
            return
        
        # Calculate required instruction
        instruction = self._calculate_instruction(current_pos, next_waypoint)
        self._send_navigation_instruction(instruction)
    
    def _calculate_instruction(self, current_pos: ContinuousPosition, waypoint: Waypoint) -> NavigationInstruction:
        """Calculate navigation instruction to reach the waypoint"""
        current_point = (current_pos.x, current_pos.y)
        target_point = (waypoint.x, waypoint.y)
        
        # Calculate vector to target
        dx = target_point[0] - current_point[0]
        dy = target_point[1] - current_point[1]
        
        # Calculate desired heading
        desired_heading = math.degrees(math.atan2(dx, dy))
        if desired_heading < 0:
            desired_heading += 360
        
        # Calculate current heading (0-360 degrees)
        current_heading = current_pos.yaw % 360
        
        # Calculate angle difference (-180 to 180)
        angle_diff = (desired_heading - current_heading + 180) % 360 - 180
        
        # Calculate distance
        distance = math.hypot(dx, dy)
        
        if abs(angle_diff) > TURN_THRESHOLD_DEG:
            # Need to turn
            direction = "left" if angle_diff < 0 else "right"
            instruction_text = f"Turn {direction} by {abs(angle_diff):.0f}° then move {distance:.1f}m"
            
            return NavigationInstruction(
                command="turn",
                direction=direction,
                angle=abs(angle_diff),
                distance=distance,
                next_waypoint=waypoint,
                instruction_text=instruction_text
            )
        else:
            # Go straight
            instruction_text = f"Move straight for {distance:.1f}m"
            
            return NavigationInstruction(
                command="move",
                direction="straight",
                distance=distance,
                next_waypoint=waypoint,
                instruction_text=instruction_text
            )

    def _get_environment_context(self, current_point: Tuple[float, float], target_point: Tuple[float, float]) -> str:
        """Generate environmental context for navigation instructions"""
        context_parts = []

        # Check if we're approaching the door (Porte)
        if self.destination and self.destination.poi_id == 142:  # Porte ID from GeoJSON
            distance_to_door = math.hypot(
                self.destination.x - current_point[0],
                self.destination.y - current_point[1]
            )
            if distance_to_door < 3:
                context_parts.append("toward the door")

        # Check for nearby tables or obstacles
        for poi, props in self.map.pois:
            if props.get('id') == 142:  # Skip door as we already checked it
                continue
                
            poi_point = poi[0] if isinstance(poi, tuple) else poi
            distance = math.hypot(poi_point[0]-current_point[0], poi_point[1]-current_point[1])
            
            if distance < 2:  # Only consider nearby POIs
                poi_name = props.get('name', 'object')
                if "table" in poi_name.lower():
                    context_parts.append(f"past the {poi_name}")
                elif "bureau" in poi_name.lower():
                    context_parts.append(f"near the {poi_name}")

        # Check if we're near walls
        for wall, _ in self.map.walls:
            if isinstance(wall[0], list):  # Polygon wall
                wall_points = wall[0]
            else:  # LineString wall
                wall_points = wall
                
            for i in range(len(wall_points)-1):
                segment_start = wall_points[i]
                segment_end = wall_points[i+1]
                distance = self._distance_point_to_line_segment(current_point, segment_start, segment_end)
                if distance < 1:
                    context_parts.append("along the wall")
                    break

        # Combine context parts
        if context_parts:
            return ", ".join(context_parts)
        return "through the open area"
    
    def _send_navigation_instruction(self, instruction: NavigationInstruction):
        """Send navigation instruction to mobile app via MQTT and print to console"""
        instruction_data = {
            'command': instruction.command,
            'direction': instruction.direction,
            'distance': instruction.distance,
            'angle': instruction.angle
        }
        
        if instruction.next_waypoint:
            instruction_data['next_waypoint'] = {
                'x': instruction.next_waypoint.x,
                'y': instruction.next_waypoint.y,
                'is_poi': instruction.next_waypoint.is_poi,
                'poi_id': instruction.next_waypoint.poi_id,
                'poi_name': instruction.next_waypoint.poi_name
            }
        
        # Print navigation instruction to console
        print("=" * 50)
        print("NAVIGATION INSTRUCTION:")
        
        if instruction.command == "turn":
            print(f"🔄 TURN {instruction.direction.upper()}")
            print(f"   Angle: {instruction.angle:.1f}°")
            if instruction.distance:
                print(f"   Distance to target: {instruction.distance:.2f}m")
        
        elif instruction.command == "move":
            print(f"➡️  MOVE {instruction.direction.upper()}")
            if instruction.distance:
                print(f"   Distance: {instruction.distance:.2f}m")
        
        elif instruction.command == "arrived":
            print("🎯 DESTINATION REACHED!")
            if instruction.next_waypoint and instruction.next_waypoint.poi_name:
                print(f"   You have arrived at: {instruction.next_waypoint.poi_name}")
        
        # Print waypoint information if available
        if instruction.next_waypoint and instruction.command != "arrived":
            if instruction.next_waypoint.is_poi and instruction.next_waypoint.poi_name:
                print(f"   Target: {instruction.next_waypoint.poi_name}")
            else:
                print(f"   Target coordinates: ({instruction.next_waypoint.x:.2f}, {instruction.next_waypoint.y:.2f})")
        
        print("=" * 50)
        
        # Send via MQTT as before
        try:
            self.mqtt_client.publish(
                MQTT_TOPIC_INSTRUCTIONS,
                json.dumps(instruction_data)
            )
        except Exception as e:
            print(f"Failed to send navigation instruction: {e}")


def main():
    # Load GeoJSON map
    with open('indoor_map.geojson') as f:
        geojson_data = json.load(f)
    
    # Create and start navigation system
    nav_system = IndoorNavigationSystem(geojson_data)
    if not nav_system.initialize():
        print("Failed to initialize navigation system")
        return
    
    try:
        # Automatically set destination to Porte (ID 142)
        poi = nav_system.map.find_poi_by_id(142)  # Porte ID from GeoJSON
        if not poi:
            print("Porte (ID 142) not found in map")
            return
            
        nav_system.destination = Waypoint(
            x=poi[0][0],
            y=poi[0][1],
            is_poi=True,
            poi_id=142,
            poi_name=poi[1].get('name', 'Porte')
        )
        
        print(f"\nNavigating to: {nav_system.destination.poi_name}")
        
        # Plan initial path
        nav_system._plan_path_to_destination()
        
        # Start navigation
        nav_system.start_navigation()
        
        # Wait until arrival
        while nav_system.destination is not None:
            time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping navigation system...")
    finally:
        nav_system.stop_navigation()

if __name__ == "__main__":
    main()