"""
Spatial Visualization Module for Request Generation and Vehicle Movement Analysis
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd

class SpatialVisualization:
    """Visualize spatial patterns of requests and vehicle movements"""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        
    def plot_request_and_vehicle_distribution(self, env, save_path=None, adp_value=None):
        """
        Create scatter plots showing:
        1. Request generation hotspots
        2. Vehicle most frequently visited locations
        3. Charging station locations
        
        Args:
            env: Environment object
            save_path: Path to save the plot
            adp_value: ADP value for title
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 构建标题后缀
        title_suffix = f" (ADP={adp_value})" if adp_value is not None else ""
        
        # Plot 1: Request Generation Hotspots
        if hasattr(env, 'request_generation_history') and env.request_generation_history:
            pickup_coords = [req['pickup_coords'] for req in env.request_generation_history]
            dropoff_coords = [req['dropoff_coords'] for req in env.request_generation_history]
            hotspot_indices = [req['hotspot_idx'] for req in env.request_generation_history]
            
            pickup_x = [coord[0] for coord in pickup_coords]
            pickup_y = [coord[1] for coord in pickup_coords]
            dropoff_x = [coord[0] for coord in dropoff_coords]
            dropoff_y = [coord[1] for coord in dropoff_coords]
            
            # Color by hotspot
            colors = ['red', 'blue', 'green']
            hotspot_colors = [colors[idx] for idx in hotspot_indices]
            
            ax1.scatter(pickup_x, pickup_y, c=hotspot_colors, alpha=0.6, s=30, label='Pickup locations')
            ax1.scatter(dropoff_x, dropoff_y, c='orange', alpha=0.3, s=20, marker='x', label='Dropoff locations')
            
            # Mark hotspot centers
            hotspots = [
                (self.grid_size // 4, self.grid_size // 4),
                (3 * self.grid_size // 4, self.grid_size // 4),
                (self.grid_size // 2, 3 * self.grid_size // 4)
            ]
            for i, (hx, hy) in enumerate(hotspots):
                ax1.scatter(hx, hy, c=colors[i], s=200, marker='*', 
                           edgecolors='black', linewidth=2, label=f'Hotspot {i+1}')
            
            ax1.set_xlim(0, self.grid_size-1)
            ax1.set_ylim(0, self.grid_size-1)
            ax1.set_title(f'Request Generation Distribution{title_suffix}\n(Pickup & Dropoff Locations)', fontsize=12)
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Vehicle Most Frequently Visited Locations
        if hasattr(env, 'vehicle_position_history') and env.vehicle_position_history:
            all_positions = []
            vehicle_types = []
            
            for vehicle_id, history in env.vehicle_position_history.items():
                vehicle_type = env.vehicles[vehicle_id]['type']
                for pos_record in history:
                    all_positions.append(pos_record['coords'])
                    vehicle_types.append(vehicle_type)
            
            if all_positions:
                # Count frequency of each position
                position_counts = Counter(all_positions)
                
                # Separate by vehicle type
                ev_positions = [pos for pos, vtype in zip(all_positions, vehicle_types) if vtype == 'EV']
                aev_positions = [pos for pos, vtype in zip(all_positions, vehicle_types) if vtype == 'AEV']
                
                if ev_positions:
                    ev_x = [pos[0] for pos in ev_positions]
                    ev_y = [pos[1] for pos in ev_positions]
                    ax2.scatter(ev_x, ev_y, c='red', alpha=0.4, s=20, label='EV positions')
                
                if aev_positions:
                    aev_x = [pos[0] for pos in aev_positions]
                    aev_y = [pos[1] for pos in aev_positions]
                    ax2.scatter(aev_x, aev_y, c='blue', alpha=0.4, s=20, label='AEV positions')
                
                # Highlight most frequent positions
                most_frequent = position_counts.most_common(10)
                if most_frequent:
                    freq_x = [pos[0] for pos, count in most_frequent]
                    freq_y = [pos[1] for pos, count in most_frequent]
                    # Reduce circle size: use count * 3 instead of count * 10, cap at 100
                    freq_sizes = [min(count * 3, 100) for pos, count in most_frequent]
                    
                    ax2.scatter(freq_x, freq_y, c='yellow', s=freq_sizes, 
                               alpha=0.8, edgecolors='black', linewidth=1,
                               label='High frequency locations')
            
            ax2.set_xlim(0, self.grid_size-1)
            ax2.set_ylim(0, self.grid_size-1)
            ax2.set_title('Vehicle Movement Patterns\n(Most Frequently Visited Locations)', fontsize=12)
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Charging Station Distribution and Utilization
        if hasattr(env, 'charging_manager') and env.charging_manager.stations:
            station_coords = []
            station_utilizations = []
            
            for station in env.charging_manager.stations.values():
                x = station.location % self.grid_size
                y = station.location // self.grid_size
                station_coords.append((x, y))
                
                # Calculate utilization
                utilization = len(station.current_vehicles) / station.max_capacity
                station_utilizations.append(utilization)
            
            if station_coords:
                station_x = [coord[0] for coord in station_coords]
                station_y = [coord[1] for coord in station_coords]
                
                # Size by utilization
                station_sizes = [max(50, util * 300) for util in station_utilizations]
                colors_util = plt.cm.RdYlBu_r(station_utilizations)
                
                scatter = ax3.scatter(station_x, station_y, c=colors_util, s=station_sizes,
                                    alpha=0.8, edgecolors='black', linewidth=2)
                
                # Add colorbar for utilization
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label('Station Utilization Rate', rotation=270, labelpad=15)
                
                # Add station IDs as labels
                for i, (x, y) in enumerate(station_coords):
                    ax3.annotate(f'S{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, fontweight='bold')
            
            ax3.set_xlim(0, self.grid_size-1)
            ax3.set_ylim(0, self.grid_size-1)
            ax3.set_title(f'Charging Station Distribution{title_suffix}\n(Size = Utilization Rate)', fontsize=12)
            ax3.set_xlabel('X Coordinate')
            ax3.set_ylabel('Y Coordinate')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Spatial distribution plot saved to: {save_path}")
        
        return fig
    
    def analyze_spatial_patterns(self, env, adp_value=None):
        """
        Analyze spatial patterns and return statistics
        
        Args:
            env: Environment object
            adp_value: ADP value for analysis context
        """
        analysis = {
            'request_patterns': {},
            'vehicle_patterns': {},
            'station_patterns': {},
            'adp_value': adp_value
        }
        
        # Analyze request patterns
        if hasattr(env, 'request_generation_history') and env.request_generation_history:
            hotspot_counts = Counter([req['hotspot_idx'] for req in env.request_generation_history])
            pickup_coords = [req['pickup_coords'] for req in env.request_generation_history]
            
            analysis['request_patterns'] = {
                'total_requests': len(env.request_generation_history),
                'hotspot_distribution': dict(hotspot_counts),
                'pickup_spread': {
                    'x_range': (min(coord[0] for coord in pickup_coords), 
                              max(coord[0] for coord in pickup_coords)),
                    'y_range': (min(coord[1] for coord in pickup_coords), 
                              max(coord[1] for coord in pickup_coords))
                }
            }
        
        # Analyze vehicle patterns
        if hasattr(env, 'vehicle_position_history') and env.vehicle_position_history:
            all_positions = []
            for history in env.vehicle_position_history.values():
                all_positions.extend([record['coords'] for record in history])
            
            if all_positions:
                position_counts = Counter(all_positions)
                most_visited = position_counts.most_common(5)
                
                analysis['vehicle_patterns'] = {
                    'total_movements': len(all_positions),
                    'unique_positions': len(position_counts),
                    'most_visited_locations': most_visited,
                    'coverage_ratio': len(position_counts) / (self.grid_size * self.grid_size)
                }
        
        # Analyze station patterns
        if hasattr(env, 'charging_manager') and env.charging_manager.stations:
            station_utilizations = []
            for station in env.charging_manager.stations.values():
                utilization = len(station.current_vehicles) / station.max_capacity
                station_utilizations.append(utilization)
            
            analysis['station_patterns'] = {
                'total_stations': len(env.charging_manager.stations),
                'avg_utilization': np.mean(station_utilizations),
                'max_utilization': max(station_utilizations),
                'min_utilization': min(station_utilizations)
            }
        
        return analysis
    
    def print_spatial_analysis(self, analysis):
        """Print spatial analysis results"""
        adp_info = f" (ADP={analysis.get('adp_value', 'N/A')})" if analysis.get('adp_value') is not None else ""
        
        print("\n" + "="*60)
        print(f"📍 SPATIAL PATTERN ANALYSIS{adp_info}")
        print("="*60)
        
        # Request patterns
        if analysis['request_patterns']:
            req_patterns = analysis['request_patterns']
            print(f"\n🎯 REQUEST GENERATION PATTERNS:")
            print(f"   Total requests generated: {req_patterns['total_requests']}")
            if 'hotspot_distribution' in req_patterns:
                print(f"   Hotspot distribution:")
                for hotspot_idx, count in req_patterns['hotspot_distribution'].items():
                    percentage = (count / req_patterns['total_requests']) * 100
                    # Handle None hotspot_idx
                    hotspot_display = "Unknown" if hotspot_idx is None else f"{hotspot_idx + 1}"
                    print(f"     Hotspot {hotspot_display}: {count} requests ({percentage:.1f}%)")
        
        # Vehicle patterns
        if analysis['vehicle_patterns']:
            veh_patterns = analysis['vehicle_patterns']
            print(f"\n🚗 VEHICLE MOVEMENT PATTERNS:")
            print(f"   Total vehicle movements: {veh_patterns['total_movements']}")
            print(f"   Unique positions visited: {veh_patterns['unique_positions']}")
            print(f"   Grid coverage ratio: {veh_patterns['coverage_ratio']:.2%}")
            if 'most_visited_locations' in veh_patterns:
                print(f"   Top 5 most visited locations:")
                for i, (coords, count) in enumerate(veh_patterns['most_visited_locations']):
                    print(f"     {i+1}. {coords}: {count} visits")
        
        # Station patterns
        if analysis['station_patterns']:
            station_patterns = analysis['station_patterns']
            print(f"\n🔋 CHARGING STATION PATTERNS:")
            print(f"   Total stations: {station_patterns['total_stations']}")
            print(f"   Average utilization: {station_patterns['avg_utilization']:.1%}")
            print(f"   Utilization range: {station_patterns['min_utilization']:.1%} - {station_patterns['max_utilization']:.1%}")
    
    def create_comprehensive_spatial_plot(self, env, save_path, adpvalue=None, demand_pattern="unknown"):
        """
        Create comprehensive spatial analysis plot (4 subplots)
        Args:
            env: Environment object
            save_path: Save path
            adpvalue: ADP value
            demand_pattern: Demand pattern
        Returns:
            bool: Whether the image was successfully generated
        """
        print(f"🗺️ Generating spatial visualization...")
        
        try:
            from collections import Counter
            import matplotlib.pyplot as plt
            
            # Create 4-subplot layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Request generation heatmap
            print(f"   📊 Plotting request generation heatmap...")
            if hasattr(env, 'request_generation_history') and env.request_generation_history:
                pickup_coords = [req['pickup_coords'] for req in env.request_generation_history]
                pickup_x = [coord[0] for coord in pickup_coords]
                pickup_y = [coord[1] for coord in pickup_coords]
                
                ax1.hist2d(pickup_x, pickup_y, bins=self.grid_size//2, alpha=0.7, cmap='Reds')
                ax1.set_title(f'Request Generation Heatmap (ADP={adpvalue})\n(Pattern: {demand_pattern})')
                ax1.set_xlabel('X Coordinate')
                ax1.set_ylabel('Y Coordinate')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No request generation data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f'Request Generation Heatmap (ADP={adpvalue})\n(No data)')
            
            # Plot 2: Vehicle historical visit patterns
            print(f"   🚗 Analyzing vehicle visit patterns...")
            if hasattr(env, 'vehicle_position_history') and env.vehicle_position_history:
                all_positions = []
                vehicle_types = []
                
                for vehicle_id, history in env.vehicle_position_history.items():
                    if vehicle_id in env.vehicles:
                        vehicle_type = env.vehicles[vehicle_id]['type']
                        for pos_record in history:
                            all_positions.append(pos_record['coords'])
                            vehicle_types.append(vehicle_type)
                
                if all_positions:
                    # Count position visit frequency
                    position_counts = Counter(all_positions)
                    
                    # Classify by vehicle type
                    ev_positions = [pos for pos, vtype in zip(all_positions, vehicle_types) if vtype == 'EV']
                    aev_positions = [pos for pos, vtype in zip(all_positions, vehicle_types) if vtype == 'AEV']
                    
                    if ev_positions:
                        ev_x = [pos[0] for pos in ev_positions]
                        ev_y = [pos[1] for pos in ev_positions]
                        ax2.scatter(ev_x, ev_y, c='red', alpha=0.4, s=20, label='EV visits')
                    
                    if aev_positions:
                        aev_x = [pos[0] for pos in aev_positions]
                        aev_y = [pos[1] for pos in aev_positions]
                        ax2.scatter(aev_x, aev_y, c='blue', alpha=0.4, s=20, label='AEV visits')
                    
                    # Highlight high-frequency locations with smaller circles
                    most_frequent = position_counts.most_common(10)
                    if most_frequent:
                        freq_x = [pos[0] for pos, count in most_frequent]
                        freq_y = [pos[1] for pos, count in most_frequent]
                        # Reduce circle size: use count * 3 instead of count * 10
                        freq_sizes = [min(count * 3, 100) for pos, count in most_frequent]  # Cap at 100
                        
                        ax2.scatter(freq_x, freq_y, c='yellow', s=freq_sizes, 
                                   alpha=0.8, edgecolors='black', linewidth=1,
                                   label='High frequency locations')
                    
                    print(f"      ✓ Vehicle visit data: total={len(all_positions)}, unique locations={len(position_counts)}")
                    ax2.set_title(f'Vehicle Movement Patterns (ADP={adpvalue})\n(Most Frequently Visited Locations)')
                else:
                    print(f"      ⚠️ No vehicle historical visit data, showing final positions...")
                    # Show final vehicle distribution as fallback
                    self._plot_final_vehicle_positions(env, ax2, adpvalue)
            else:
                print(f"      ⚠️ Environment missing vehicle_position_history attribute")
                self._plot_final_vehicle_positions(env, ax2, adpvalue)
            
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Charging station utilization
            print(f"   🔋 Analyzing charging station utilization...")
            if hasattr(env, 'charging_manager') and env.charging_manager.stations:
                station_data = []
                for station_id, station in env.charging_manager.stations.items():
                    station_x = station.location % self.grid_size
                    station_y = station.location // self.grid_size
                    utilization = len(station.current_vehicles) / station.max_capacity if station.max_capacity > 0 else 0
                    station_data.append((station_x, station_y, utilization))
                
                if station_data:
                    station_x, station_y, utilizations = zip(*station_data)
                    scatter = ax3.scatter(station_x, station_y, c=utilizations, s=200, 
                                        cmap='YlOrRd', alpha=0.8, edgecolor='black')
                    plt.colorbar(scatter, ax=ax3, label='Utilization Rate')
                    ax3.set_title(f'Charging Station Utilization (ADP={adpvalue})')
                    ax3.set_xlabel('X Coordinate')
                    ax3.set_ylabel('Y Coordinate')
                    ax3.grid(True, alpha=0.3)
                    print(f"      ✓ Displayed {len(station_data)} charging stations")
                else:
                    ax3.text(0.5, 0.5, 'No charging station data', ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title(f'Charging Station Utilization (ADP={adpvalue})\n(No data)')
            else:
                ax3.text(0.5, 0.5, 'No charging manager', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title(f'Charging Station Utilization (ADP={adpvalue})\n(No data)')
            
            # Plot 4: Performance summary text
            print(f"   📈 Generating performance summary...")
            ax4.axis('off')
            self._create_performance_summary(env, ax4, adpvalue, demand_pattern)
            
            # Save image
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Spatial visualization saved: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Spatial visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _plot_final_vehicle_positions(self, env, ax, adpvalue):
        """Plot final vehicle position distribution (fallback)"""
        vehicle_positions = [v['coordinates'] for v in env.vehicles.values()]
        if vehicle_positions:
            ev_x = [pos[0] for v_id, v in env.vehicles.items() if v['type'] == 'EV' for pos in [v['coordinates']]]
            ev_y = [pos[1] for v_id, v in env.vehicles.items() if v['type'] == 'EV' for pos in [v['coordinates']]]
            aev_x = [pos[0] for v_id, v in env.vehicles.items() if v['type'] == 'AEV' for pos in [v['coordinates']]]
            aev_y = [pos[1] for v_id, v in env.vehicles.items() if v['type'] == 'AEV' for pos in [v['coordinates']]]
            
            ax.scatter(ev_x, ev_y, c='blue', alpha=0.6, label='EV final positions', s=50)
            ax.scatter(aev_x, aev_y, c='green', alpha=0.6, label='AEV final positions', s=50)
            ax.set_title(f'Final Vehicle Distribution (ADP={adpvalue})\n(No historical data)')
        else:
            ax.text(0.5, 0.5, 'No vehicle data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Vehicle Distribution (ADP={adpvalue})\n(No data)')
    
    def _create_performance_summary(self, env, ax, adpvalue, demand_pattern):
        """创建性能摘要文本"""
        try:
            # 获取环境统计信息
            stats = env.get_stats() if hasattr(env, 'get_stats') else {}
            
            # 构建摘要文本
            summary_text = f"""
Performance Summary
ADP Value: {adpvalue}
Demand Pattern: {demand_pattern}

Grid Size: {self.grid_size}x{self.grid_size}
Total Vehicles: {env.num_vehicles if hasattr(env, 'num_vehicles') else 'N/A'}
Charging Stations: {env.num_stations if hasattr(env, 'num_stations') else 'N/A'}

Active Requests: {stats.get('active_requests', 'N/A')}
Completed Requests: {stats.get('completed_requests', 'N/A')}
Average Battery Level: {stats.get('average_battery', 0):.2f}

EV Vehicles: {sum(1 for v in env.vehicles.values() if v['type'] == 'EV') if hasattr(env, 'vehicles') else 'N/A'}
AEV Vehicles: {sum(1 for v in env.vehicles.values() if v['type'] == 'AEV') if hasattr(env, 'vehicles') else 'N/A'}
            """
            
            ax.text(0.1, 0.9, summary_text.strip(), transform=ax.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f'Unable to generate performance summary\nError: {str(e)}', 
                    ha='center', va='center', transform=ax.transAxes)
