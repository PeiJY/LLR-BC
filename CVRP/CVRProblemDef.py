
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import random

distribution_parameters = {
    'U':[],
    'GM':[5,50],
    # 'R':[0.2,180], # distribution R is not used in this setting, instead, using C: compression
    'E':[0.3,40],
    'C':[0.3,0.1], # tube width, gaussion var
    'Grid':[0.2,0.8], # min width/height ratio, max width/height ratio
    'Ring':[0.3,0.4,0.05,0.2,0.8], # min ring radius, max ring radius, var of the distance to the torus
}

demand_distribution_parameters = {
    'U':[1,10], # min demand, max demand. uniformly random demand
    'GM':[1,10], # min demand, max demand. demand linear to the distance to the center
    'E':[5,1,1,10], # mean, var, min demand, max demand. 1 - gaussain demand
    'C':[5,1,1,10], # mean, var, min demand, max demand. gaussain demand
    'Grid':[1,10,1], # min demand, max demand. linear to the distance to the depot
    'Ring':[1,10,1], # min demand, max demand. 1 - linear to the distance to the depot
}

def min_max_normalize(problem):
    min_vals = problem.min(dim=0).values
    max_vals = problem.max(dim=0).values
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1 
    normalized_problem = (problem - min_vals) / range_vals
    return normalized_problem

def get_random_node_xy(batch_size, problem_size, distribution, distribution_params=None):
    if distribution in distribution_parameters.keys():
        if distribution_params is None: 
            distribution_params = distribution_parameters[distribution]
        return sample_instances(batch_size,problem_size,distribution,distribution_params)
    elif distribution == 'MIX':
        distributions = distribution_parameters.keys()
        depot_xys=[]
        node_xys=[]
        node_demands=[]
        for _ in range(batch_size):
            d = random.choice(distributions)
            depot_xy, node_xy, node_demand = sample_instances(1, problem_size, distribution=d,parameters=distribution_parameters[d])
            depot_xy = depot_xy.squeeze(0)
            node_xy = node_xy.squeeze(0)
            node_demand = node_demand.squeeze(0)
            depot_xys.append(depot_xy)
            node_xys.append(node_xy)
            node_demands.append(node_demand)
    else:
        raise ValueError('problem distribution {} not exist'.format(distribution))
    depot_xys=torch.stack(depot_xys)
    node_xys=torch.stack(node_xys)
    node_demands=torch.stack(node_demands)
    return depot_xys,node_xys,node_demands  # Shape: (batch_size, problem_size, 2)


def sample_instances(batch_size, problem_size, distribution, parameters):
    # parameters = distribution_params[distribution]
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    if distribution == 'U':
        # uniform demand
        min_demand = demand_distribution_parameters[distribution][0]
        max_demand = demand_distribution_parameters[distribution][1]
        node_xy = torch.rand(size=(batch_size, problem_size, 2))
        node_demand = torch.randint(min_demand, max_demand+1, size=(batch_size, problem_size)) 
        # shape: (batch, problem, 2)
    elif distribution == 'GM':
        # demand related to the distance to the center
        min_demand = demand_distribution_parameters[distribution][0]
        max_demand = demand_distribution_parameters[distribution][1]
        node_xy = []
        node_demand = []
        for b in range(batch_size):
            c = parameters[0]
            l = parameters[1]
            centers = torch.rand(c,2) * l
            problem = centers.clone()
            total_num = 0
            demands = [torch.randint(low=min_demand, high=max_demand+1, size=(c,))]
            for i in range(c):
                if i == c-1:
                    num = problem_size - c - total_num
                else:
                    num = int(problem_size / c) -1
                    total_num += num
                xy_offset = torch.randn(num, 2)
                coordinates = centers[i] + xy_offset
                distances_to_center = [(x**2 + y**2)**0.5 for x, y in xy_offset]
                min_d = min(distances_to_center)
                max_d = max(distances_to_center)
                min_demand = demand_distribution_parameters[distribution][0]
                max_demand = demand_distribution_parameters[distribution][1]
                demands.append(torch.tensor([int((d - min_d) / (max_d - min_d)*(max_demand-min_demand))+min_demand for d in distances_to_center]))
                problem = torch.cat((problem,coordinates),dim=0)
            problem = min_max_normalize(problem)
            demands = torch.cat(demands,dim=0)
            node_xy.append(problem)
            node_demand.append(demands)
        node_xy = torch.stack(node_xy)
        node_demand = torch.stack(node_demand)
    elif distribution == 'C':
        # 1- Gaussain demand
        tube_width = distribution_parameters[distribution][0]
        tube_relocate_var = distribution_parameters[distribution][1]
        mean = demand_distribution_parameters[distribution][0]
        var = demand_distribution_parameters[distribution][1]
        min_demand = demand_distribution_parameters[distribution][2]
        max_demand = demand_distribution_parameters[distribution][3]
        node_demand = torch.randn(size=(batch_size, problem_size)) * var + mean # generate Gaussain denmands in float
        node_demand = node_demand.round().int() # convert to int
        node_demand = torch.clamp(node_demand,min=min_demand,max=max_demand)
        node_demand = max_demand - node_demand
        node_xy = []
        for b in range(batch_size):
            problem = torch.rand(size=(problem_size, 2))
            p1 = torch.rand(2)
            p2 = torch.rand(2)
            d = p2 - p1                           # [2]
            d_norm = torch.norm(d)                
            direction = d / d_norm                # [2]
            w = problem - p1                       # [N,2]
            distances = torch.abs(d[0]*w[:,1] - d[1]*w[:,0]) / d_norm  # [N]
            
            mask = distances < tube_width            
            if not mask.any():
                print("no node have distance smaller than {}, skip".format(tube_width))
                # continue
            else:
                sel_points = problem[mask]              
                w_sel = w[mask]                           
                proj_lens = (w_sel * direction).sum(dim=1, keepdim=True)    
                q = p1 + proj_lens * direction                                 
                taus = torch.randn(sel_points.size(0), 1) * (tube_relocate_var)     
                taus = torch.clamp(torch.abs(taus), max=1.0)                
                pi_prime = q - (sel_points - q) * taus                      
                problem[mask] = pi_prime
            # problem = min_max_normalize(problem)
            problem.clamp_(0.0, 1.0)
            node_xy.append(problem)
        node_xy = torch.stack(node_xy)
    elif distribution == 'E':
        # Gaussain demand
        mean = demand_distribution_parameters[distribution][0]
        var = demand_distribution_parameters[distribution][1]
        min_demand = demand_distribution_parameters[distribution][2]
        max_demand = demand_distribution_parameters[distribution][3]
        node_demand = torch.randn(size=(batch_size, problem_size)) * var + mean # generate Gaussain denmands in float
        node_demand = node_demand.round() # convert to interger
        node_demand = torch.clamp(node_demand,min=min_demand,max=max_demand)
        node_xy = []
        for b in range(batch_size):
            problem = torch.rand(size=(problem_size, 2))
            explosion_center = torch.rand(2)
            R = parameters[0]
            lambda_exp = parameters[1]
            distances = torch.norm(problem - explosion_center, dim=1)
            within_radius_mask = distances <= R
            if within_radius_mask.any():
                s = torch.distributions.Exponential(lambda_exp).sample((within_radius_mask.sum(),))
                direction_vectors = problem[within_radius_mask] - explosion_center
                normalized_directions = direction_vectors / distances[within_radius_mask].unsqueeze(1)
                problem[within_radius_mask] = explosion_center + (R + s).unsqueeze(1) * normalized_directions
            # problem = min_max_normalize(problem)
            problem.clamp_(0.0, 1.0)
            node_xy.append(problem)
        node_xy = torch.stack(node_xy)
    elif distribution == 'Grid':
        node_xy_list = []
        node_demand_list = []
        min_ratio = distribution_parameters[distribution][0]
        max_ratio = distribution_parameters[distribution][1]
        min_demand = demand_distribution_parameters[distribution][0]
        max_demand = demand_distribution_parameters[distribution][1]
        noise_range = demand_distribution_parameters[distribution][2]
        for b in range(batch_size):
            ratio = torch.empty(1).uniform_(min_ratio, max_ratio).item()
            if random.random() > 0.5:
                w = 1.0 - 1e-5
                h = ratio
            else:
                h = 1.0 - 1e-5
                w = ratio
            cx = torch.empty(1).uniform_(w/2, 1 - w/2).item()
            cy = torch.empty(1).uniform_(h/2, 1 - h/2).item()

            side_x = math.ceil(math.sqrt(problem_size * (w / h)))
            side_y = math.ceil(problem_size / side_x)
            x_vals = torch.linspace(-w/2, w/2, steps=side_x)
            y_vals = torch.linspace(-h/2, h/2, steps=side_y)
            yy, xx = torch.meshgrid(y_vals, x_vals, indexing='ij')
            coords = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
            coords = coords[:problem_size]
            coords += torch.tensor([cx, cy])  
            
            coords = coords.clamp(0.0, 1.0)
            
            depot = depot_xy[b, 0]  # shape: (2,)
            distances = torch.norm(coords - depot, dim=1)  # shape: (problem_size,)
            noise = torch.rand(problem_size) * noise_range
            demand_raw = distances + noise


            norm = (demand_raw - demand_raw.min()) / (demand_raw.max() - demand_raw.min() + 1e-8)
            demand = norm * (max_demand - min_demand) + min_demand
            
            demand = demand.round().int() # convert to int
            demand = torch.clamp(demand,min=min_demand,max=max_demand)

            node_xy_list.append(coords)
            node_demand_list.append(demand)

        node_xy = torch.stack(node_xy_list, dim=0)         # (batch_size, problem_size, 2)
        node_demand = torch.stack(node_demand_list, dim=0) # (batch_size, problem_size)
    elif distribution == 'Ring':
        center = torch.tensor([0.5, 0.5])  
        node_xy_list = []
        node_demand_list = []
        ring_params = []
        radius_min = distribution_parameters[distribution][0]
        radius_max = distribution_parameters[distribution][1]
        sigma = distribution_parameters[distribution][2]
        min_ratio = distribution_parameters[distribution][3]
        max_ratio = distribution_parameters[distribution][4]
        r_mean_range = (radius_min,radius_max)
        min_demand = demand_distribution_parameters[distribution][0]
        max_demand = demand_distribution_parameters[distribution][1]
        noise_range = demand_distribution_parameters[distribution][2]
        for b in range(batch_size):
            r_mean = torch.empty(1).uniform_(*r_mean_range).item()
            angles = torch.rand(problem_size) * 2 * math.pi
            radial_noise = torch.randn(problem_size) * sigma
            radii = r_mean + radial_noise
            x = center[0] + radii * torch.cos(angles)
            y = center[1] + radii * torch.sin(angles)
            coords = torch.stack([x, y], dim=-1)
            
            ratio = torch.empty(1).uniform_(0.2, 0.8).item()
            if random.random()>0.5:
                scale_x = 1.0
                scale_y = ratio
            else:
                scale_x = ratio
                scale_y = 1.0
            coords[:, 0] *= scale_x
            coords[:, 1] *= scale_y

            coords = coords.clamp(0.0, 1.0) 

            node_xy_list.append(coords)
            ring_params.append((r_mean, sigma))
            depot = depot_xy[b, 0]  # shape: (2,)
            distances = torch.norm(coords - depot, dim=1)  # shape: (problem_size,)
            noise = torch.rand(problem_size) * noise_range
            demand_raw = distances + noise


            # norm = (demand_raw - demand_raw.min()) / (demand_raw.max() - demand_raw.min() + 1e-8)
            norm = 1- (demand_raw - demand_raw.min()) / (demand_raw.max() - demand_raw.min() + 1e-8)
            demand = norm * (max_demand - min_demand) + min_demand
            
            demand = demand.round().int() # convert to int
            demand = torch.clamp(demand,min=min_demand,max=max_demand)

            node_demand_list.append(demand)
        node_xy = torch.stack(node_xy_list, dim=0)  # (batch_size, problem_size, 2)
        node_demand = torch.stack(node_demand_list, dim=0) # (batch_size, problem_size)
    else:
        raise ValueError('distribution {} not exist.'.format(distribution))
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        # raise NotImplementedError
        demand_scaler = 100

    node_demand = node_demand.float() / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

    
def draw_cvrp_instance(depot_xy, node_xy, node_demand):

    batch_size, problem_size, _ = node_xy.shape
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        # raise NotImplementedError
        demand_scaler = 100
    if batch_size == 1:
        plt.figure(figsize=(4, 4))
        depot = depot_xy[0].squeeze(0).numpy() 
        nodes = node_xy[0].numpy()  
        demands = node_demand[0].numpy() * demand_scaler
        
        scatter = plt.scatter(nodes[:, 0], nodes[:, 1], c=demands, cmap='viridis', s=50, label='node')
        
        plt.scatter(depot[0], depot[1], c='red', s=100, marker='s', label='depot')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Demand')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
    else:
        row_size = 2
        batch_size, problem_size, _ = node_xy.shape
        n_rows = (batch_size + 3) // row_size
        fig, axes = plt.subplots(n_rows, min(batch_size, row_size), figsize=(15, 15))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
    
        for b in range(batch_size):
            row = b // row_size
            col = b % row_size
            ax = axes[row, col]
            ax.set_ylim(0,1)
            ax.set_xlim(0,1)
            depot = depot_xy[b].squeeze(0).numpy() 
            nodes = node_xy[b].numpy()  
            demands = node_demand[b].numpy() 
            
            scatter = ax.scatter(nodes[:, 0], nodes[:, 1], c=demands, cmap='viridis', s=50, label='node')
            
            ax.scatter(depot[0], depot[1], c='red', s=100, marker='s', label='depot')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Demand')
            
            for i, (x, y) in enumerate(nodes):
                ax.text(x, y, f'{demands[i]:.2f}', fontsize=8, ha='right')
            
            ax.set_title(f'Instance {b + 1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True)
            ax.legend()
        
        for b in range(batch_size, n_rows * row_size):
            row = b // row_size
            col = b % row_size
            fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    
    sizes = {'U':20,'Ring':20,'E':50,'Grid':50,'GM':100,'C':100}
    batch_size = 1

    for d in sizes.keys():
        ps = distribution_parameters[d]
        depot_xy, node_xy, node_demand = get_random_node_xy(batch_size, sizes[d], distribution=d, distribution_params=ps)
        print(depot_xy.shape,node_xy.shape,node_demand.shape)
        draw_cvrp_instance(depot_xy, node_xy, node_demand)
        file_name = 'CVRP_{}-{}_{}.pdf'.format(d,ps,sizes[d])
        print('save to:',file_name)
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.savefig(file_name)