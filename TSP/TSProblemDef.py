
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


def min_max_normalize(problem):
    min_vals = problem.min(dim=0).values
    max_vals = problem.max(dim=0).values
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1 
    normalized_problem = (problem - min_vals) / range_vals
    return normalized_problem

def get_random_problems(batch_size, problem_size, distribution, distribution_params=None):
    
    if distribution in distribution_parameters.keys():
        if distribution_params is None: 
            distribution_params = distribution_parameters[distribution]
        return sample_instances(batch_size,problem_size,distribution,distribution_params)
    elif distribution == 'MIX':
        distributions = distribution_parameters.keys()
        problems = []
        for _ in range(batch_size):
            d = random.choice(distributions)
            problem = sample_instances(1, problem_size, distribution=d,parameters=distribution_parameters[d]).squeeze(0)
            problems.append(problem)
    else:
        raise ValueError('problem distribution {} not exist'.format(distribution))
    return torch.stack(problems)  # Shape: (batch_size, problem_size, 2)
    
def sample_instances(batch_size, problem_size, distribution, parameters):
    # parameters=distribution_params[distribution]
    if distribution == 'U':
        problems = torch.rand(size=(batch_size, problem_size, 2))
    elif distribution == 'GM':
        problems = []
        for b in range(batch_size):
            c = parameters[0]
            l = parameters[1]
            centers = torch.rand(c,2) * l
            problem = centers.clone()
            total_num = 0
            for i in range(c):
                if i == c-1:
                    num = problem_size - c - total_num
                else:
                    num = int(problem_size / c) -1
                    total_num += num
                coordinates = centers[i] + torch.randn(num, 2)
                problem = torch.cat((problem,coordinates),dim=0)
            problem = min_max_normalize(problem)
            problems.append(problem)
        problems = torch.stack(problems)
    elif distribution == 'C':
        # 1- Gaussain demand
        tube_width = distribution_parameters[distribution][0]
        tube_relocate_var = distribution_parameters[distribution][1]
        problems = []
        for b in range(batch_size):
            problem = torch.rand(size=(problem_size, 2))
            # explosion_center = torch.rand(2)
            p1 = torch.rand(2)
            p2 = torch.rand(2)
            d = p2 - p1                           # [2]
            d_norm = torch.norm(d)                # 
            direction = d / d_norm                # [2]
            w = problem - p1                       # [N,2]
            distances = torch.abs(d[0]*w[:,1] - d[1]*w[:,0]) / d_norm  # [N]
            
            mask = distances < tube_width               # [N] 
            if not mask.any():
                print("no node have distance smaller than {}, skip".format(tube_width))
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
            problems.append(problem)
        problems = torch.stack(problems)
    elif distribution == 'E':
        problems = []
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
            problem = min_max_normalize(problem)
            problems.append(problem)
        problems = torch.stack(problems)
    elif distribution == 'Grid':
        problems = []
        min_ratio = distribution_parameters[distribution][0]
        max_ratio = distribution_parameters[distribution][1]
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
            problems.append(coords)
        problems = torch.stack(problems, dim=0)         # (batch_size, problem_size, 2)
    elif distribution == 'Ring':
        center = torch.tensor([0.5, 0.5]) 
        problems = []
        ring_params = []
        radius_min = distribution_parameters[distribution][0]
        radius_max = distribution_parameters[distribution][1]
        sigma = distribution_parameters[distribution][2]
        min_ratio = distribution_parameters[distribution][3]
        max_ratio = distribution_parameters[distribution][4]
        r_mean_range = (radius_min,radius_max)
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

            problems.append(coords)
        problems = torch.stack(problems, dim=0)  # (batch_size, problem_size, 2)
    else:
        raise ValueError('distribution {} not exist.'.format(distribution))
    # problems.shape: (batch, problem, 2)
    return problems


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems



def draw_tsp_problems(problems):
    """
    draw TSP problem instance
    :param problems: a (batch_size, problem_size, 2) tensor
    """
    batch_size, problem_size, _ = problems.shape
    if batch_size == 1:
        plt.figure(figsize=(4, 4))
        problem = problems[0].numpy()
        x = problem[:, 0]
        y = problem[:, 1]
        
        plt.scatter(x, y, c='blue', s=50)
        # plt.title(f'Problem {b + 1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
    else:
        n_rows = (batch_size + 3) // 4
        fig, axes = plt.subplots(n_rows, min(batch_size, 4), figsize=(15, 4 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for b in range(batch_size):
            row = b // 4
            col = b % 4
            ax = axes[row, col]
            
            problem = problems[b].numpy()
            x = problem[:, 0]
            y = problem[:, 1]
            
            ax.scatter(x, y, c='blue', s=50)
            ax.set_title(f'Problem {b + 1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True)
        
        for b in range(batch_size, n_rows * 4):
            row = b // 4
            col = b % 4
            fig.delaxes(axes[row, col])
        
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    
    sizes = {'U':20,'Ring':20,'E':50,'Grid':50,'GM':100,'C':100}
    batch_size = 1
    for d in sizes.keys():
        problem_size = sizes[d]
        ps = distribution_parameters[d]
        problems = get_random_problems(batch_size, problem_size, distribution=d, distribution_params=ps)
        draw_tsp_problems(problems)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig('./TSP_{}_{}.pdf'.format(d,problem_size))
