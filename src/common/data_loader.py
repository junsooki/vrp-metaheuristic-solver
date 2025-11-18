"""
Data loading utilities
- Generate random VRP instances
- Parse standard VRP file formats (CVRPLIB, Solomon, etc.)
"""

import random
import os
from .problem import Customer, Depot, VRPInstance


def generate_random_instance(num_customers: int, capacity: float = None, 
                             grid_size: int = 100, seed: int = None):
    """
    Generate random CVRP instance
    
    Args:
        num_customers: Number of customers
        capacity: Vehicle capacity (default: num_customers * 2)
        grid_size: Size of coordinate grid
        seed: Random seed
    
    Returns:
        VRPInstance
    """
    if seed is not None:
        random.seed(seed)
    
    if capacity is None:
        capacity = num_customers * 2
    
    # Generate depot at center
    depot = Depot(x=grid_size / 2, y=grid_size / 2)
    
    # Generate customers
    customers = []
    for i in range(1, num_customers + 1):
        customer = Customer(
            id=i,
            x=random.uniform(0, grid_size),
            y=random.uniform(0, grid_size),
            demand=random.uniform(1, 10)
        )
        customers.append(customer)
    
    name = f"random_n{num_customers}_c{int(capacity)}_s{seed}"
    
    return VRPInstance(name, depot, customers, capacity)


def load_vrp_instance(filepath: str):
    """
    Load VRP instance from file (auto-detects format)
    
    Supported formats:
    - Solomon format (.txt)
    - CVRPLIB/TSPLIB format (.vrp)
    - Simple text format
    
    Args:
        filepath: Path to instance file
    
    Returns:
        VRPInstance
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instance file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Detect format
    if any('CVRP' in line for line in lines[:5]):
        return _parse_cvrplib_format(filepath, lines)
    elif any('VEHICLE' in line.upper() for line in lines[:10]):
        return _parse_solomon_format(filepath, lines)
    else:
        return _parse_simple_format(filepath, lines)


def load_instances_from_folder(folder_path: str, extensions=None):
    """
    Load all VRP instances from a folder
    
    Args:
        folder_path: Path to folder containing instance files
        extensions: List of file extensions to load (default: ['.txt', '.vrp'])
    
    Returns:
        List of VRPInstance objects
    """
    if extensions is None:
        extensions = ['.txt', '.vrp', '.dat']
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a folder: {folder_path}")
    
    instances = []
    files_found = []
    
    # Get all files with matching extensions
    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
        
        # Check extension
        _, ext = os.path.splitext(filename)
        if ext.lower() not in extensions:
            continue
        
        files_found.append(filename)
        
        try:
            instance = load_vrp_instance(filepath)
            instances.append(instance)
            print(f"✓ Loaded: {filename} ({instance.num_customers} customers)")
        except Exception as e:
            print(f"✗ Failed to load {filename}: {str(e)}")
    
    if not files_found:
        print(f"⚠️  No instance files found in {folder_path}")
        print(f"   Looking for extensions: {extensions}")
    
    print(f"\nTotal loaded: {len(instances)}/{len(files_found)} instances")
    
    return instances


def _parse_solomon_format(filepath: str, lines: list):
    """
    Parse Solomon VRPTW benchmark format
    
    Format:
    - Line 0: Instance name
    - Line 4: num_vehicles capacity
    - Line 9+: customer data (id, x, y, demand, ready_time, due_time, service_time)
    """
    name = lines[0] if lines[0] else os.path.basename(filepath)
    
    # Find vehicle info line
    vehicle_line = None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            vehicle_line = i
            break
    
    if vehicle_line is None:
        raise ValueError("Could not find vehicle information")
    
    parts = lines[vehicle_line].split()
    num_vehicles = int(parts[0])
    capacity = float(parts[1])
    
    # Find start of customer data (usually line with "CUST NO." or similar)
    data_start = vehicle_line + 1
    for i in range(vehicle_line, len(lines)):
        if 'CUST' in lines[i].upper() or 'CUSTOMER' in lines[i].upper():
            data_start = i + 1
            break
    
    # Parse customers
    depot_data = None
    customers = []
    
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        
        try:
            cust_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            demand = float(parts[3])
            ready_time = float(parts[4])
            due_time = float(parts[5])
            service_time = float(parts[6])
            
            if cust_id == 0:
                # Depot
                depot_data = {
                    'x': x, 'y': y,
                    'tw_start': ready_time,
                    'tw_end': due_time
                }
            else:
                customer = Customer(
                    id=cust_id,
                    x=x, y=y,
                    demand=demand,
                    service_time=service_time,
                    tw_start=ready_time,
                    tw_end=due_time
                )
                customers.append(customer)
        except (ValueError, IndexError):
            continue
    
    if depot_data is None:
        raise ValueError("Depot not found in instance file")
    
    depot = Depot(
        x=depot_data['x'],
        y=depot_data['y'],
        tw_start=depot_data['tw_start'],
        tw_end=depot_data['tw_end']
    )
    
    return VRPInstance(name, depot, customers, capacity, num_vehicles)


def _parse_cvrplib_format(filepath: str, lines: list):
    """
    Parse CVRPLIB/TSPLIB format
    
    Format:
    NAME: instance_name
    COMMENT: ...
    TYPE: CVRP
    DIMENSION: n
    EDGE_WEIGHT_TYPE: EUC_2D
    CAPACITY: capacity
    NODE_COORD_SECTION
    1 x y
    2 x y
    ...
    DEMAND_SECTION
    1 demand
    2 demand
    ...
    DEPOT_SECTION
    1
    -1
    """
    name = None
    dimension = None
    capacity = None
    
    # Parse header
    for line in lines:
        if line.startswith('NAME'):
            name = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            capacity = float(line.split(':')[1].strip())
    
    if name is None:
        name = os.path.basename(filepath)
    
    # Find sections
    coord_start = None
    demand_start = None
    depot_start = None
    
    for i, line in enumerate(lines):
        if 'NODE_COORD_SECTION' in line:
            coord_start = i + 1
        elif 'DEMAND_SECTION' in line:
            demand_start = i + 1
        elif 'DEPOT_SECTION' in line:
            depot_start = i + 1
    
    # Parse coordinates
    coords = {}
    if coord_start:
        for i in range(coord_start, len(lines)):
            parts = lines[i].split()
            if len(parts) != 3:
                break
            try:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[node_id] = (x, y)
            except ValueError:
                break
    
    # Parse demands
    demands = {}
    if demand_start:
        for i in range(demand_start, len(lines)):
            parts = lines[i].split()
            if len(parts) != 2:
                break
            try:
                node_id = int(parts[0])
                demand = float(parts[1])
                demands[node_id] = demand
            except ValueError:
                break
    
    # Create depot (node 1 is depot)
    depot_x, depot_y = coords.get(1, (0, 0))
    depot = Depot(x=depot_x, y=depot_y)
    
    # Create customers
    customers = []
    for node_id in range(2, dimension + 1):
        if node_id in coords:
            x, y = coords[node_id]
            demand = demands.get(node_id, 0)
            customer = Customer(
                id=node_id - 1,  # Reindex from 1
                x=x, y=y,
                demand=demand
            )
            customers.append(customer)
    
    return VRPInstance(name, depot, customers, capacity)


def _parse_simple_format(filepath: str, lines: list):
    """
    Parse simple text format
    
    Format:
    name
    num_customers capacity
    depot_x depot_y
    customer1_x customer1_y demand1
    customer2_x customer2_y demand2
    ...
    """
    name = lines[0] if lines else os.path.basename(filepath)
    
    parts = lines[1].split()
    num_customers = int(parts[0])
    capacity = float(parts[1])
    
    depot_parts = lines[2].split()
    depot = Depot(x=float(depot_parts[0]), y=float(depot_parts[1]))
    
    customers = []
    for i in range(3, 3 + num_customers):
        parts = lines[i].split()
        customer = Customer(
            id=i - 2,
            x=float(parts[0]),
            y=float(parts[1]),
            demand=float(parts[2])
        )
        customers.append(customer)
    
    return VRPInstance(name, depot, customers, capacity)
