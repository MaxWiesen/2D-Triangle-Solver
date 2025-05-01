import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_nodes(folder = 'inputs/HW 7 Input Files', ext='R'):
    with open(f'{folder}/nodes{ext}', 'r') as f:
        node = {packet[0] - 1: packet[1:] for packet in pd.read_csv(f, sep='\s+', skiprows=1, names=['node', 'x1', 'x2'], dtype={'node': int, 'x1': float, 'x2': float}).to_dict('split')['data']}

    ndim = dof_per_node = 2
    ndofs = dof_per_node * len(node)
    
    gcon = {}
    for i in range(len(node)):
        gcon[i] = {}
        for j in range(dof_per_node):
            gcon[i][j + 1] = dof_per_node * i + j + 1

    return ndim, node, dof_per_node, gcon, ndofs


def load_elements(node, ndim, folder = 'inputs/HW 7 Input Files', ext='R'):
    with open(f'{folder}/elements{ext}', 'r') as f:
        nele, E, v = f.readline().split()
        nele = int(nele)
        E = float(E)
        v = float(v)    
        elements = pd.read_csv(f, sep=r'\s+', names=['elenum', 'elenodes1', 'elenodes2', 'elenodes3'], dtype={'elenum': int, 'elenodes1': int, 'elenodes2': int, 'elenodes3': int}).set_index('elenum')
        elements.index -= 1
        elements.elenodes1 -= 1
        elements.elenodes2 -= 1
        elements.elenodes3 -= 1
        nodes_per_ele = len(elements.columns)
    
    # Calculate area for each element
    A = np.zeros(len(elements))
    bele = np.zeros((nele, 3, nodes_per_ele * ndim))
    for i in range(nele):
        # Get coordinates of the three nodes
        x1, y1 = node[elements.elenodes1.values[i]]
        x2, y2 = node[elements.elenodes2.values[i]]
        x3, y3 = node[elements.elenodes3.values[i]]
        
        # Calculate area using cross product method
        A[i] = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) / 2
        b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
        c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
        bele[i] = np.array([[b1, 0, b2, 0, b3, 0], 
                            [0, c1, 0, c2, 0, c3], 
                            [c1, b1, c2, b2, c3, b3]]) / 2 / A[i]

    C = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])

    return nele, nodes_per_ele, [elements.elenodes1.values, elements.elenodes2.values, elements.elenodes3.values], E, v, bele, A, C


def load_forces(folder = 'inputs/HW 7 Input Files', ext='R'):
    with open(f'{folder}/forces{ext}', 'r') as f:
        nfbcs = int(f.readline())
        forces = pd.read_csv(f, sep=r'\s+', names=['node', 'dof', 'value'], dtype={'node': int, 'dof': int, 'value': float})
    forces.node -= 1

    return nfbcs, forces.node.values, forces.dof.values, forces.value.values


def load_displacements(folder = 'inputs/HW 7 Input Files', ext='R'):
    with open(f'{folder}/displacements{ext}', 'r') as f:
        ndbcs = int(f.readline())
        displacements = pd.read_csv(f, sep=r'\s+', names=['node', 'dof', 'value'], dtype={'node': int, 'dof': int, 'value': float})
    displacements.node -= 1

    return ndbcs, displacements.node.values, displacements.dof.values, displacements.value.values


def plot_mesh(node, elenodes, centroids=False, reflect_full_plate=False):
    # Set the base font settings
    plt.rcParams['font.family'] = 'Times New Roman'
    
    fig, ax = plt.subplots()
    
    def plot_quadrant(x_coords, y_coords, sign_x=1, sign_y=1):
        transformed_x = [x * sign_x for x in x_coords]
        transformed_y = [y * sign_y for y in y_coords]
        ax.scatter(transformed_x, transformed_y, c='blue', s=5, label='Nodes' if sign_x == 1 and sign_y == 1 else "")
        
        for i in range(len(elenodes[0])):
            node1 = elenodes[0][i]
            node2 = elenodes[1][i]
            node3 = elenodes[2][i]
            
            x = [node[node1][0] * sign_x, node[node2][0] * sign_x, node[node3][0] * sign_x, node[node1][0] * sign_x]
            y = [node[node1][1] * sign_y, node[node2][1] * sign_y, node[node3][1] * sign_y, node[node1][1] * sign_y]
            ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.5)
            
            if centroids:
                centroid_x = sign_x * (node[node1][0] + node[node2][0] + node[node3][0]) / 3
                centroid_y = sign_y * (node[node1][1] + node[node2][1] + node[node3][1]) / 3
                ax.plot(centroid_x, centroid_y, 'rx', markersize=3, 
                       label='Centroids' if i == 0 and sign_x == 1 and sign_y == 1 else "", alpha=0.35)
        
        # Add critical points markers
        if sign_x == 1 and sign_y == 1:
            # Plot (1,0) critical point
            ax.scatter([1 * sign_x], [0], c='green', marker='s', s=50, label='Critical Points')
            # Plot (0,1) critical point
            ax.scatter([0], [1 * sign_y], c='green', marker='s', s=50)
    
    # Get base coordinates
    x_coords = [node[i][0] for i in range(len(node))]
    y_coords = [node[i][1] for i in range(len(node))]
    
    if reflect_full_plate:
        # Plot all four quadrants
        plot_quadrant(x_coords, y_coords, 1, 1)   # Quadrant 1
        plot_quadrant(x_coords, y_coords, -1, 1)  # Quadrant 2
        plot_quadrant(x_coords, y_coords, -1, -1) # Quadrant 3
        plot_quadrant(x_coords, y_coords, 1, -1)  # Quadrant 4
        ax.set_title('Full Plate with Hole Mesh Visualization', fontfamily='Times New Roman', fontsize=16)
    else:
        # Original single quadrant plot
        plot_quadrant(x_coords, y_coords)
        ax.set_title('Triangular Mesh Visualization', fontfamily='Times New Roman', fontsize=16)
    
    ax.set_xlabel('X', fontfamily='Times New Roman', fontsize=14)
    ax.set_ylabel('Y', fontfamily='Times New Roman', fontsize=14)
    ax.grid()
    if centroids or not reflect_full_plate:
        ax.legend(prop={'family': 'Times New Roman', 'size': 12})
    ax.tick_params(labelsize=12)
    ax.axis('equal')  # Make the plot aspect ratio 1:1
    plt.show()


def plot_field_on_mesh(node, elenodes, field_values, title, cmap='coolwarm', reflect_full_plate=False, use_deformed=False, displacements=None, scale_factor=1.0):
    """
    Plot a field on the mesh, with option to show on deformed configuration.
    
    Args:
        node: Dictionary of node coordinates
        elenodes: List of arrays containing node indices for each element
        field_values: Array of field values for each element
        title: Title of the plot
        cmap: Colormap to use for the heatmap
        reflect_full_plate: If True, reflect the mesh to show full plate with hole
        use_deformed: If True, plot on deformed configuration
        displacements: Array of nodal displacements (required if use_deformed=True)
        scale_factor: Factor to scale displacements for visualization (default=1.0)
    """
    # Set the base font settings
    plt.rcParams['font.family'] = 'Times New Roman'
    
    fig, ax = plt.subplots()
    
    def get_deformed_coords(node_idx, sign_x=1, sign_y=1):
        """Helper function to get deformed coordinates of a node"""
        if use_deformed and displacements is not None:
            x = node[node_idx][0] + scale_factor * displacements[node_idx][0]
            y = node[node_idx][1] + scale_factor * displacements[node_idx][1]
        else:
            x = node[node_idx][0]
            y = node[node_idx][1]
        return sign_x * x, sign_y * y
    
    def plot_quadrant(field_data, sign_x=1, sign_y=1):
        # Get node coordinates (original or deformed)
        nodes_x = []
        nodes_y = []
        for i in range(len(node)):
            x, y = get_deformed_coords(i, sign_x, sign_y)
            nodes_x.append(x)
            nodes_y.append(y)
        
        # Create triangle connectivity list
        tri_connect = [[elenodes[0][i], elenodes[1][i], elenodes[2][i]] for i in range(len(elenodes[0]))]
        
        # Create the colored triangles
        collection = plt.tripcolor(
            nodes_x, nodes_y,
            tri_connect,
            field_data,
            cmap=cmap,
            shading='flat'
        )
        
        # Plot mesh edges
        for i in range(len(elenodes[0])):
            node1 = elenodes[0][i]
            node2 = elenodes[1][i]
            node3 = elenodes[2][i]
            
            x1, y1 = get_deformed_coords(node1, sign_x, sign_y)
            x2, y2 = get_deformed_coords(node2, sign_x, sign_y)
            x3, y3 = get_deformed_coords(node3, sign_x, sign_y)
            
            x = [x1, x2, x3, x1]
            y = [y1, y2, y3, y1]
            ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
        
        return collection
    
    # Calculate field data for elements
    field_data = field_values
    
    if reflect_full_plate:
        # Plot all four quadrants
        collection = plot_quadrant(field_data, 1, 1)    # Quadrant 1
        plot_quadrant(field_data, -1, 1)   # Quadrant 2
        plot_quadrant(field_data, -1, -1)  # Quadrant 3
        plot_quadrant(field_data, 1, -1)   # Quadrant 4
        base_title = f'{title} Distribution\nFull Plate with Hole'
        title_fontsize = 14  # Smaller font for longer title
    else:
        # Original single quadrant plot
        collection = plot_quadrant(field_data)
        base_title = f'{title} Distribution'
        title_fontsize = 16
    
    if use_deformed:
        base_title = f'{base_title}\nDeformed Configuration (Scale = {scale_factor:.3g})'
        title_fontsize = max(12, title_fontsize - 2)  # Reduce font size for additional line
    
    ax.set_title(base_title, fontfamily='Times New Roman', fontsize=title_fontsize, pad=15)
    
    # Add colorbar with proper font
    cbar = plt.colorbar(collection, label=title)
    cbar.ax.set_ylabel(title, fontfamily='Times New Roman', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    ax.set_xlabel('X', fontfamily='Times New Roman', fontsize=14)
    ax.set_ylabel('Y', fontfamily='Times New Roman', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    ax.axis('equal')  # Make the plot aspect ratio 1:1
    
    plt.tight_layout()  # Adjust layout to prevent title overlap
    plt.show()


def calculate_scf_at_critical_points(node, elenodes, stress_vm, stress):
    """
    Calculate stress concentration factors at critical points (0,1) and (1,0)
    by finding nearest elements and performing linear extrapolation.
    
    Args:
        node: Dictionary of node coordinates
        elenodes: List of arrays containing node indices for each element
        stress_vm: Array of von Mises stress values for each element
        stress: Array of stress components [σxx, σyy, τxy] for each element
    
    Returns:
        dict: Dictionary containing SCF values and intermediate calculations
    """
    # Initialize containers for closest elements
    right_elements = []  # Elements near (1,0)
    top_elements = []    # Elements near (0,1)
    
    # Calculate centroids and find closest elements
    for i in range(len(elenodes[0])):
        node1 = elenodes[0][i]
        node2 = elenodes[1][i]
        node3 = elenodes[2][i]
        
        # Calculate centroid
        centroid_x = (node[node1][0] + node[node2][0] + node[node3][0]) / 3
        centroid_y = (node[node1][1] + node[node2][1] + node[node3][1]) / 3
        
        # Store element info with distance from axis
        element_info = {
            'index': i,
            'centroid': (centroid_x, centroid_y),
            'stress_vm': stress_vm[i],
            'stress_xx': stress[i, 0],
            'stress_yy': stress[i, 1],
            'dist_from_axis': 0
        }
        
        # Check if element is near right side (x ≈ 1)
        if centroid_x > 0.5:  # Only consider elements in right half
            element_info['dist_from_axis'] = abs(centroid_y)  # Distance from x-axis
            right_elements.append(element_info)
            
        # Check if element is near top side (y ≈ 1)
        if centroid_y > 0.5:  # Only consider elements in top half
            element_info = element_info.copy()  # Create new dict to avoid reference issues
            element_info['dist_from_axis'] = abs(centroid_x)  # Distance from y-axis
            top_elements.append(element_info)
    
    # Sort by distance from axis and take two closest
    right_elements.sort(key=lambda x: x['dist_from_axis'])
    top_elements.sort(key=lambda x: x['dist_from_axis'])
    
    right_closest = right_elements[:2]
    top_closest = top_elements[:2]
    
    # Linear extrapolation for right side (1,0)
    x1, y1 = right_closest[0]['centroid']
    x2, y2 = right_closest[1]['centroid']
    
    # Extrapolate each stress component for right side
    stress_xx_1 = right_closest[0]['stress_xx']
    stress_xx_2 = right_closest[1]['stress_xx']
    stress_yy_1 = right_closest[0]['stress_yy']
    stress_yy_2 = right_closest[1]['stress_yy']
    
    # Use x-coordinates for extrapolation to (1,0)
    if abs(x1 - x2) > 1e-10:  # Avoid division by zero
        slope_right_xx = (stress_xx_2 - stress_xx_1) / (x2 - x1)
        slope_right_yy = (stress_yy_2 - stress_yy_1) / (x2 - x1)
        stress_xx_at_1_0 = stress_xx_1 + slope_right_xx * (1 - x1)
        stress_yy_at_1_0 = stress_yy_1 + slope_right_yy * (1 - x1)
    else:
        stress_xx_at_1_0 = (stress_xx_1 + stress_xx_2) / 2
        stress_yy_at_1_0 = (stress_yy_1 + stress_yy_2) / 2
    
    # Linear extrapolation for top side (0,1)
    x1, y1 = top_closest[0]['centroid']
    x2, y2 = top_closest[1]['centroid']
    
    # Extrapolate each stress component for top side
    stress_xx_1 = top_closest[0]['stress_xx']
    stress_xx_2 = top_closest[1]['stress_xx']
    stress_yy_1 = top_closest[0]['stress_yy']
    stress_yy_2 = top_closest[1]['stress_yy']
    
    # Use y-coordinates for extrapolation to (0,1)
    if abs(y1 - y2) > 1e-10:  # Avoid division by zero
        slope_top_xx = (stress_xx_2 - stress_xx_1) / (y2 - y1)
        slope_top_yy = (stress_yy_2 - stress_yy_1) / (y2 - y1)
        stress_xx_at_0_1 = stress_xx_1 + slope_top_xx * (1 - y1)
        stress_yy_at_0_1 = stress_yy_1 + slope_top_yy * (1 - y1)
    else:
        stress_xx_at_0_1 = (stress_xx_1 + stress_xx_2) / 2
        stress_yy_at_0_1 = (stress_yy_1 + stress_yy_2) / 2
    
    results = {
        'right_elements': right_closest,
        'top_elements': top_closest,
        'stress_at_1_0': {
            'xx': stress_xx_at_1_0,
            'yy': stress_yy_at_1_0
        },
        'stress_at_0_1': {
            'xx': stress_xx_at_0_1,
            'yy': stress_yy_at_0_1
        }
    }
    
    return results


def plot_edge_stress_distribution(node, elenodes, scf_results, stress, verbose=False):
    # Set the base font settings
    plt.rcParams['font.family'] = 'Times New Roman'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initialize containers for edge elements
    x_edge_elements = []  # Elements along y=0
    y_edge_elements = []  # Elements along x=0
    
    tolerance = 1e-6  # Stricter tolerance for being "on" the axis
    
    # Find elements with at least two nodes on the axes
    for i in range(len(elenodes[0])):
        nodes = [elenodes[0][i], elenodes[1][i], elenodes[2][i]]
        node_coords = [(node[n][0], node[n][1]) for n in nodes]
        
        # Count nodes on each axis
        nodes_on_x_axis = sum(1 for _, y in node_coords if abs(y) < tolerance)
        nodes_on_y_axis = sum(1 for x, _ in node_coords if abs(x) < tolerance)
        
        # Calculate centroid
        centroid_x = sum(x for x, _ in node_coords) / 3
        centroid_y = sum(y for _, y in node_coords) / 3
        
        element_info = {
            'index': i,
            'centroid': (centroid_x, centroid_y),
            'stress_xx': stress[i, 0],  # σxx component
            'stress_yy': stress[i, 1],  # σyy component
            'nodes': nodes,
            'node_coords': node_coords
        }
        
        # Add elements with at least 2 nodes on an axis
        if nodes_on_x_axis >= 2:
            x_edge_elements.append(element_info)
        if nodes_on_y_axis >= 2:
            y_edge_elements.append(element_info)
    
    # Sort elements by their position along the axis
    x_edge_elements.sort(key=lambda x: x['centroid'][0])
    y_edge_elements.sort(key=lambda x: x['centroid'][1])
    
    # Plot stress along x-axis (y=0)
    if x_edge_elements:
        x_coords = [elem['centroid'][0] for elem in x_edge_elements]
        x_stress_xx = [elem['stress_xx'] for elem in x_edge_elements]
        x_stress_yy = [elem['stress_yy'] for elem in x_edge_elements]
        
        ax1.plot(x_coords, x_stress_xx, 'b.-', label='σxx')
        ax1.plot(x_coords, x_stress_yy, 'r.-', label='σyy')
        
        # Add extrapolated points with smaller markers
        ax1.scatter(1, scf_results['stress_at_1_0']['xx'], c='blue', marker='s', s=40, label='Extrapolated σxx')
        ax1.scatter(1, scf_results['stress_at_1_0']['yy'], c='red', marker='s', s=40, label='Extrapolated σyy')
    
    ax1.set_xlabel('x coordinate', fontfamily='Times New Roman', fontsize=14)
    ax1.set_ylabel('Normal Stress Components', fontfamily='Times New Roman', fontsize=14)
    ax1.set_title('Stress Distribution along y=0', fontfamily='Times New Roman', fontsize=16)
    ax1.grid()
    ax1.legend(prop={'family': 'Times New Roman', 'size': 12})
    ax1.tick_params(labelsize=12)
    
    # Plot stress along y-axis (x=0)
    if y_edge_elements:
        y_coords = [elem['centroid'][1] for elem in y_edge_elements]
        y_stress_xx = [elem['stress_xx'] for elem in y_edge_elements]
        y_stress_yy = [elem['stress_yy'] for elem in y_edge_elements]
        
        ax2.plot(y_coords, y_stress_xx, 'b.-', label='σxx')
        ax2.plot(y_coords, y_stress_yy, 'r.-', label='σyy')
        
        # Add extrapolated points with smaller markers
        ax2.scatter(1, scf_results['stress_at_0_1']['xx'], c='blue', marker='s', s=40, label='Extrapolated σxx')
        ax2.scatter(1, scf_results['stress_at_0_1']['yy'], c='red', marker='s', s=40, label='Extrapolated σyy')
    
    ax2.set_xlabel('y coordinate', fontfamily='Times New Roman', fontsize=14)
    ax2.set_ylabel('Normal Stress Components', fontfamily='Times New Roman', fontsize=14)
    ax2.set_title('Stress Distribution along x=0', fontfamily='Times New Roman', fontsize=16)
    ax2.grid()
    ax2.legend(prop={'family': 'Times New Roman', 'size': 12})
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    if verbose:
        # Print the data points and node information
        print("\nStress Distribution Data Points:")
        
        print("\nAlong y=0 axis:")
        for elem in x_edge_elements:
            print(f"\nElement at x = {elem['centroid'][0]:.4f}:")
            print(f"  σxx = {elem['stress_xx']:.4f}")
            print(f"  σyy = {elem['stress_yy']:.4f}")
            print("  Node coordinates:")
            for i, (x, y) in enumerate(elem['node_coords']):
                print(f"    Node {i+1}: ({x:.4f}, {y:.4f})")
        
        print("\nAlong x=0 axis:")
        for elem in y_edge_elements:
            print(f"\nElement at y = {elem['centroid'][1]:.4f}:")
            print(f"  σxx = {elem['stress_xx']:.4f}")
            print(f"  σyy = {elem['stress_yy']:.4f}")
            print("  Node coordinates:")
            for i, (x, y) in enumerate(elem['node_coords']):
                print(f"    Node {i+1}: ({x:.4f}, {y:.4f})")


def main(ext='24', show_individual_plots=False):
    """
    Process a single mesh file and return relevant data for analysis.
    
    Args:
        ext (str): Extension/identifier for the mesh files
        show_individual_plots (bool): Whether to show individual mesh plots and analysis
        
    Returns:
        tuple: (node, elenodes, stress, u) containing mesh and analysis results
    """
    t = 1

    ndim, node, dof_per_node, gcon, ndofs = load_nodes(folder='inputs/HW 7 Input Files', ext=ext)
    nele, nodes_per_ele, elenodes, E, v, bele, A, C = load_elements(node, ndim, folder='inputs/HW 7 Input Files', ext=ext)
    nfbcs, forcenode, forcedof, forcevalue = load_forces(folder='inputs/HW 7 Input Files', ext=ext)
    ndbcs, dispnode, dispdof, dispvalue = load_displacements(folder='inputs/HW 7 Input Files', ext=ext)

    if show_individual_plots:
        plot_mesh(node, elenodes, centroids=True, reflect_full_plate=False)

    for i in range(ndbcs):
        bcdof = gcon[dispnode[i]][dispdof[i]]
        for j in range(len(node)):
            for k in range(dof_per_node):
                if gcon[j][k + 1] > bcdof:
                    gcon[j][k + 1] -= 1
        gcon[dispnode[i]][dispdof[i]] = len(node) * dof_per_node
        ndofs -= 1

    K_red = np.zeros((ndofs, ndofs))
    F_red = np.zeros(ndofs)
    u = np.zeros((len(node), ndim))

    for i in range(nfbcs):
        dof = gcon[forcenode[i]][forcedof[i]]
        F_red[dof - 1] += forcevalue[i]

    for i in range(ndbcs):
        u[dispnode[i]][dispdof[i] - 1] = dispvalue[i]

    for iele in range(nele):
        kele = A[iele] * (bele[iele].T @ C @ bele[iele])
        for inode in range(nodes_per_ele):
            for idof in range(dof_per_node):
                idoflocal = inode * dof_per_node + idof
                idofglobal = gcon[elenodes[inode][iele]][idof + 1]
                if idofglobal <= ndofs:
                    for jnode in range(nodes_per_ele):
                        for jdof in range(dof_per_node):
                            jdoflocal = jnode * dof_per_node + jdof
                            jdofglobal = gcon[elenodes[jnode][iele]][jdof + 1]
                            if jdofglobal <= ndofs:
                                K_red[idofglobal - 1][jdofglobal - 1] += kele[idoflocal][jdoflocal]
                            else:
                                F_red[idofglobal - 1] -= kele[idoflocal][jdoflocal] * u[elenodes[jnode][iele]][jdof]

    sol = np.linalg.solve(K_red, F_red)

    for i in range(len(node)):
        for j in range(dof_per_node):
            dof = gcon[i][j + 1]
            if dof <= ndofs:
                u[i][j] = sol[dof - 1]
    
    strain = np.zeros((nele, nodes_per_ele))
    stress = np.zeros((nele, nodes_per_ele))

    for iele in range(nele):
        uele = np.zeros(nodes_per_ele * dof_per_node)
        for inode in range(nodes_per_ele):
            curr_node = elenodes[inode][iele]
            for j in range(dof_per_node):
                dof = gcon[curr_node][j + 1]
                if dof <= ndofs:
                    uele[dof_per_node * inode + j] = sol[dof - 1]
                else:
                    uele[dof_per_node * inode + j] = u[curr_node][j]
        strain[iele, :] = bele[iele] @ uele
        stress[iele, :] = C @ strain[iele, :]

    if show_individual_plots:
        # Calculate von Mises equivalent strain
        strain_vm = np.sqrt(2/3 * ((strain[:,0] - strain[:,1])**2 + strain[:,0]**2 + strain[:,1]**2 + 3/2*strain[:,2]**2))
        
        # Plot strain on both original and deformed configurations
        plot_field_on_mesh(node, elenodes, strain_vm, "Von Mises Strain", reflect_full_plate=False)
        plot_field_on_mesh(node, elenodes, strain_vm, "Von Mises Strain", reflect_full_plate=False, 
                          use_deformed=True, displacements=u, scale_factor=.1)
        
        # Calculate von Mises equivalent stress
        stress_vm = np.sqrt(stress[:,0]**2 + stress[:,1]**2 - stress[:,0]*stress[:,1] + 3*stress[:,2]**2)
        
        # Plot stress on both original and deformed configurations
        plot_field_on_mesh(node, elenodes, stress_vm, "Von Mises Stress", reflect_full_plate=False)
        plot_field_on_mesh(node, elenodes, stress_vm, "Von Mises Stress", reflect_full_plate=False, 
                          use_deformed=True, displacements=u, scale_factor=.1)
        
        # Plot reflected versions
        plot_field_on_mesh(node, elenodes, strain_vm, "Von Mises Strain", reflect_full_plate=True)
        plot_field_on_mesh(node, elenodes, strain_vm, "Von Mises Strain", reflect_full_plate=True, 
                          use_deformed=True, displacements=u, scale_factor=.1)
        
        plot_field_on_mesh(node, elenodes, stress_vm, "Von Mises Stress", reflect_full_plate=True)
        plot_field_on_mesh(node, elenodes, stress_vm, "Von Mises Stress", reflect_full_plate=True, 
                          use_deformed=True, displacements=u, scale_factor=.1)

        # Calculate and plot stress concentration factors
        scf_results = calculate_scf_at_critical_points(node, elenodes, stress_vm, stress)
        
        print(f"\nStress Concentration Analysis for {ext} element mesh:")
        print("-" * 30)
        
        print("\nExtrapolated Stresses at Critical Points:")
        print(f"At (1,0):")
        print(f"  σxx = {scf_results['stress_at_1_0']['xx']:.4f}")
        print(f"  σyy = {scf_results['stress_at_1_0']['yy']:.4f}")
        print(f"At (0,1):")
        print(f"  σxx = {scf_results['stress_at_0_1']['xx']:.4f}")
        print(f"  σyy = {scf_results['stress_at_0_1']['yy']:.4f}")
        
        # Plot individual stress distributions along edges
        plot_edge_stress_distribution(node, elenodes, scf_results, stress)
    
    return node, elenodes, stress, u


def plot_comparative_edge_stress_distributions(mesh_results, mesh_names):
    """
    Create two figures comparing stress distributions along edges for multiple meshes.
    One figure for stresses along y=0 and another for stresses along x=0.
    
    Args:
        mesh_results (list): List of tuples (node, elenodes, stress, u) for each mesh
        mesh_names (list): List of names/identifiers for each mesh
    """
    # Set the base font settings
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Create two separate figures
    # Figure 1: Stresses along y=0 (x-axis)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Figure 2: Stresses along x=0 (y-axis)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colors and markers for different meshes
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']  # Different marker shapes
    
    for mesh_idx, (node, elenodes, stress, _) in enumerate(mesh_results):
        # Initialize containers for edge elements
        x_edge_elements = []  # Elements along y=0
        y_edge_elements = []  # Elements along x=0
        
        tolerance = 1e-6  # Tolerance for being "on" the axis
        
        # Find elements with at least two nodes on the axes
        for i in range(len(elenodes[0])):
            nodes = [elenodes[0][i], elenodes[1][i], elenodes[2][i]]
            node_coords = [(node[n][0], node[n][1]) for n in nodes]
            
            # Count nodes on each axis
            nodes_on_x_axis = sum(1 for _, y in node_coords if abs(y) < tolerance)
            nodes_on_y_axis = sum(1 for x, _ in node_coords if abs(x) < tolerance)
            
            # Calculate centroid
            centroid_x = sum(x for x, _ in node_coords) / 3
            centroid_y = sum(y for _, y in node_coords) / 3
            
            element_info = {
                'index': i,
                'centroid': (centroid_x, centroid_y),
                'stress_xx': stress[i, 0],  # σxx component
                'stress_yy': stress[i, 1],  # σyy component
            }
            
            # Add elements with at least 2 nodes on an axis
            if nodes_on_x_axis >= 2:
                x_edge_elements.append(element_info)
            if nodes_on_y_axis >= 2:
                y_edge_elements.append(element_info)
        
        # Sort elements by their position along the axis
        x_edge_elements.sort(key=lambda x: x['centroid'][0])
        y_edge_elements.sort(key=lambda x: x['centroid'][1])
        
        color = colors[mesh_idx % len(colors)]
        marker = markers[mesh_idx % len(markers)]
        label = mesh_names[mesh_idx]
        
        # Plot stresses along y=0 (Figure 1)
        if x_edge_elements:
            x_coords = [elem['centroid'][0] for elem in x_edge_elements]
            x_stress_xx = [elem['stress_xx'] for elem in x_edge_elements]
            x_stress_yy = [elem['stress_yy'] for elem in x_edge_elements]
            ax1.plot(x_coords, x_stress_xx, color=color, marker=marker, linestyle='-', 
                    label=label, markersize=6, markerfacecolor='white')
            ax2.plot(x_coords, x_stress_yy, color=color, marker=marker, linestyle='-', 
                    label=label, markersize=6, markerfacecolor='white')
        
        # Plot stresses along x=0 (Figure 2)
        if y_edge_elements:
            y_coords = [elem['centroid'][1] for elem in y_edge_elements]
            y_stress_xx = [elem['stress_xx'] for elem in y_edge_elements]
            y_stress_yy = [elem['stress_yy'] for elem in y_edge_elements]
            ax3.plot(y_coords, y_stress_xx, color=color, marker=marker, linestyle='-', 
                    label=label, markersize=6, markerfacecolor='white')
            ax4.plot(y_coords, y_stress_yy, color=color, marker=marker, linestyle='-', 
                    label=label, markersize=6, markerfacecolor='white')
    
    # Configure Figure 1 (y=0)
    ax1.set_title('σxx', fontsize=14, pad=10)
    ax1.set_xlabel('x coordinate', fontsize=14)
    ax1.set_ylabel('σxx', fontsize=14)
    ax1.grid(True)
    ax1.legend(prop={'size': 12})
    ax1.tick_params(labelsize=12)
    
    ax2.set_title('σyy', fontsize=14, pad=10)
    ax2.set_xlabel('x coordinate', fontsize=14)
    ax2.set_ylabel('σyy', fontsize=14)
    ax2.grid(True)
    ax2.legend(prop={'size': 12})
    ax2.tick_params(labelsize=12)
    
    # Configure Figure 2 (x=0)
    ax3.set_title('σxx', fontsize=14, pad=10)
    ax3.set_xlabel('y coordinate', fontsize=14)
    ax3.set_ylabel('σxx', fontsize=14)
    ax3.grid(True)
    ax3.legend(prop={'size': 12})
    ax3.tick_params(labelsize=12)
    
    ax4.set_title('σyy', fontsize=14, pad=10)
    ax4.set_xlabel('y coordinate', fontsize=14)
    ax4.set_ylabel('σyy', fontsize=14)
    ax4.grid(True)
    ax4.legend(prop={'size': 12})
    ax4.tick_params(labelsize=12)
    
    # Adjust layouts
    fig1.tight_layout()
    fig2.tight_layout()
    
    # Add figure titles after tight_layout to prevent interference
    fig1.text(0.5, 0.95, 'Stress Along x-axis (y=0)', 
              horizontalalignment='center', fontsize=16, fontfamily='Times New Roman')
    fig2.text(0.5, 0.95, 'Stress Along y-axis (x=0)', 
              horizontalalignment='center', fontsize=16, fontfamily='Times New Roman')
    
    plt.show()


def print_critical_point_stresses(mesh_results, mesh_names):
    """
    Print a formatted table of stresses at critical points for all meshes.
    
    Args:
        mesh_results: List of tuples (node, elenodes, stress, u) for each mesh
        mesh_names: List of mesh names/identifiers
    """
    # Set up table headers
    print("\nStress Analysis at Critical Points")
    print("=" * 80)
    print(f"{'Mesh':^10} | {'Point (1,0)':^32} | {'Point (0,1)':^32}")
    print(f"{'':<10} | {'σxx':^15} {'σyy':^15} | {'σxx':^15} {'σyy':^15}")
    print("-" * 80)
    
    # Process each mesh
    for (node, elenodes, stress, _), mesh_name in zip(mesh_results, mesh_names):
        # Calculate von Mises stress
        stress_vm = np.sqrt(stress[:,0]**2 + stress[:,1]**2 - stress[:,0]*stress[:,1] + 3*stress[:,2]**2)
        
        # Calculate stresses at critical points
        scf_results = calculate_scf_at_critical_points(node, elenodes, stress_vm, stress)
        
        # Extract values
        stress_10_xx = scf_results['stress_at_1_0']['xx']
        stress_10_yy = scf_results['stress_at_1_0']['yy']
        stress_01_xx = scf_results['stress_at_0_1']['xx']
        stress_01_yy = scf_results['stress_at_0_1']['yy']
        
        # Print formatted row
        print(f"{mesh_name:^10} | {stress_10_xx:^15.4f} {stress_10_yy:^15.4f} | {stress_01_xx:^15.4f} {stress_01_yy:^15.4f}")
    
    print("=" * 80)


if __name__ == '__main__':
    # List of mesh extensions to process
    mesh_exts = ['R', '6', '12', '24']
    mesh_names = ['R Ext', '6 Ext', '12 Ext', '24 Ext']
    
    # Process all meshes
    mesh_results = []
    for ext in mesh_exts:
        # Set show_individual_plots=True for the last mesh only
        show_plots = (ext == mesh_exts[-1])  # Only show for the last mesh
        # result = main(ext, show_individual_plots=show_plots)
        result = main(ext)
        mesh_results.append(result)
    
    # Print critical point stresses for all meshes
    print_critical_point_stresses(mesh_results, mesh_names)
    
    # Create comparative plots
    plot_comparative_edge_stress_distributions(mesh_results, mesh_names)

