from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns
from collections import defaultdict, deque
class ChargingIntegrationVisualization:
    """Charging integration visualization class"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
    def plot_integrated_results(self, results, save_path=None):
        """Plot comprehensive charts for integrated test results"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Charging Behavior Integration Test Results Analysis', fontsize=16, fontweight='bold')
        
        # 1. Training reward curve
        rewards = results.get('episode_rewards', [])
        if rewards:
            axes[0,0].plot(rewards, linewidth=2, color='blue')
            axes[0,0].set_title('Training Reward Curve')
            axes[0,0].set_xlabel('Episode')
            axes[0,0].set_ylabel('Cumulative Reward')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add trend line
            if len(rewards) > 1:
                z = np.polyfit(range(len(rewards)), rewards, 1)
                p = np.poly1d(z)
                axes[0,0].plot(range(len(rewards)), p(range(len(rewards))), 
                             "--", color='red', alpha=0.8, label='Trend Line')
                axes[0,0].legend()
        
        # 2. ValueFunction Loss Curve
        vf_losses = results.get('value_function_losses', [])
        vf_ev_losses = results.get('value_function_ev_losses', [])
        if vf_losses or vf_ev_losses:
            if vf_losses:
                axes[0,1].plot(vf_losses, linewidth=2, color='red', label='AEV Loss')
            if vf_ev_losses:
                axes[0,1].plot(vf_ev_losses, linewidth=2, color='blue', label='EV Loss')
            axes[0,1].set_title('ValueFunction Training Loss')
            axes[0,1].set_xlabel('Episode')
            axes[0,1].set_ylabel('Loss')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()
            
            # Add smoothed line
            if len(vf_losses) > 5:
                try:
                    from scipy.ndimage import uniform_filter1d
                    smoothed = uniform_filter1d(vf_losses, size=5)
                    axes[0,1].plot(smoothed, linewidth=2, color='darkred', alpha=0.7, label='AEV Smoothed')
                    axes[0,1].legend()
                except ImportError:
                    # If scipy not available, use simple moving average
                    window = 5
                    smoothed = []
                    for i in range(len(vf_losses)):
                        start = max(0, i - window + 1)
                        smoothed.append(np.mean(vf_losses[start:i+1]))
                    axes[0,1].plot(smoothed, linewidth=2, color='darkred', alpha=0.7, label='AEV Smoothed')
                    axes[0,1].legend()
        
        # 3. Battery level changes
        battery_levels = results.get('battery_levels', [])
        if battery_levels:
            axes[0,2].plot(battery_levels, linewidth=2, color='green')
            axes[0,2].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Low Battery Alert')
            axes[0,2].set_title('Average Battery Level Changes')
            axes[0,2].set_xlabel('Episode')
            axes[0,2].set_ylabel('Battery Ratio')
            axes[0,2].grid(True, alpha=0.3)
            axes[0,2].legend()
            axes[0,2].set_ylim(0, 1)
        
        # 4. Charging event time distribution
        charging_events = results.get('charging_events', [])
        if charging_events:
            event_times = [event['time'] for event in charging_events]
            axes[1,0].hist(event_times, bins=20, alpha=0.7, color='orange')
            axes[1,0].set_title('Charging Event Time Distribution')
            axes[1,0].set_xlabel('Time Step')
            axes[1,0].set_ylabel('Number of Charging Events')
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. Charging duration distribution
        if charging_events:
            durations = [event['duration'] for event in charging_events]
            axes[1,1].hist(durations, bins=15, alpha=0.7, color='purple')
            axes[1,1].set_title('Charging Duration Distribution')
            axes[1,1].set_xlabel('Charging Time')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Charging station usage statistics
        if charging_events:
            station_counts = defaultdict(int)
            for event in charging_events:
                station_counts[event['station_id']] += 1
            
            if station_counts:
                stations = list(station_counts.keys())
                counts = list(station_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(stations)))
                
                bars = axes[1,2].bar(stations, counts, color=colors)
                axes[1,2].set_title('Charging Station Usage Statistics')
                axes[1,2].set_xlabel('Station ID')
                axes[1,2].set_ylabel('Usage Count')
                axes[1,2].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                 str(count), ha='center', va='bottom')
        
        # 7. Vehicle charging frequency analysis
        if charging_events:
            vehicle_counts = defaultdict(int)
            for event in charging_events:
                vehicle_counts[event['vehicle_id']] += 1
            
            if vehicle_counts:
                vehicles = list(vehicle_counts.keys())
                counts = list(vehicle_counts.values())
                
                axes[2,0].bar(vehicles, counts, alpha=0.7, color='cyan')
                axes[2,0].set_title('Vehicle Charging Frequency')
                axes[2,0].set_xlabel('Vehicle ID')
                axes[2,0].set_ylabel('Charging Count')
                axes[2,0].grid(True, alpha=0.3)
        
        # 8. Learning Progress (Reward vs Loss)
        if rewards and vf_losses:
            ax_twin = axes[2,1]
            ax_twin.plot(rewards, color='blue', label='Rewards')
            ax_twin.set_xlabel('Episode')
            ax_twin.set_ylabel('Reward', color='blue')
            ax_twin.tick_params(axis='y', labelcolor='blue')
            
            ax_twin2 = ax_twin.twinx()
            if vf_losses:
                ax_twin2.plot(vf_losses, color='red', label='AEV Loss')
            if vf_ev_losses:
                ax_twin2.plot(vf_ev_losses, color='blue', label='EV Loss')
            ax_twin2.set_ylabel('Loss', color='red')
            ax_twin2.tick_params(axis='y', labelcolor='red')
            
            axes[2,1].set_title('Learning Progress: Reward vs Loss')
            axes[2,1].grid(True, alpha=0.3)
        
        # 9. Performance Summary
        if battery_levels and rewards:
            final_battery = battery_levels[-1] if battery_levels else 0
            final_reward = rewards[-1] if rewards else 0
            total_charging = len(charging_events)
            
            axes[2,2].text(0.1, 0.8, f'Final Avg Battery: {final_battery:.2f}', transform=axes[2,2].transAxes, fontsize=12)
            axes[2,2].text(0.1, 0.6, f'Final Episode Reward: {final_reward:.1f}', transform=axes[2,2].transAxes, fontsize=12)
            axes[2,2].text(0.1, 0.4, f'Total Charging Events: {total_charging}', transform=axes[2,2].transAxes, fontsize=12)
            axes[2,2].text(0.1, 0.2, f'Episodes Completed: {len(rewards)}', transform=axes[2,2].transAxes, fontsize=12)
            axes[2,2].set_title('Performance Summary')
            axes[2,2].set_xlim(0, 1)
            axes[2,2].set_ylim(0, 1)
            axes[2,2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Integrated results chart saved to: {save_path}")
        
        return fig

    def plot_charging_strategy_analysis(self, results, save_path=None):
        """Plot charging strategy analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Charging Strategy Deep Analysis', fontsize=16, fontweight='bold')
        
        charging_events = results.get('charging_events', [])
        battery_levels = results.get('battery_levels', [])
        
        # 1. Charging decision timing analysis
        if charging_events and battery_levels:
            # Simulate charging decision battery threshold analysis
            low_battery_charges = len([e for e in charging_events if random.random() < 0.7])  # Simulate low battery charging
            normal_charges = len(charging_events) - low_battery_charges
            
            labels = ['Low Battery Charging', 'Preventive Charging']
            sizes = [low_battery_charges, normal_charges]
            colors = ['#ff9999', '#66b3ff']
            
            axes[0,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0,0].set_title('Charging Decision Type Distribution')
        
        # 2. Charging efficiency analysis
        if charging_events:
            # Simulate charging efficiency data
            efficiency_data = [random.uniform(0.8, 0.95) for _ in charging_events]
            axes[0,1].hist(efficiency_data, bins=15, alpha=0.7, color='lightgreen')
            axes[0,1].axvline(np.mean(efficiency_data), color='red', linestyle='--', 
                            label=f'Average Efficiency: {np.mean(efficiency_data):.2f}')
            axes[0,1].set_title('Charging Efficiency Distribution')
            axes[0,1].set_xlabel('Charging Efficiency')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Battery management performance
        if battery_levels:
            axes[1,0].plot(battery_levels, linewidth=2, label='Actual Battery')
            axes[1,0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Danger Line')
            axes[1,0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Recommended Charging Line')
            axes[1,0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Sufficient Battery Line')
            axes[1,0].set_title('Battery Management Performance')
            axes[1,0].set_xlabel('Episode')
            axes[1,0].set_ylabel('Average Battery')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_ylim(0, 1)
        
        # 4. Charging time optimization analysis
        if charging_events:
            # Analyze charging frequency by time period
            time_periods = defaultdict(int)
            for event in charging_events:
                period = event['time'] // 20  # Divide time into periods
                time_periods[period] += 1
            
            if time_periods:
                periods = list(time_periods.keys())
                counts = list(time_periods.values())
                
                axes[1,1].bar(periods, counts, alpha=0.7, color='gold')
                axes[1,1].set_title('Charging Frequency in Different Time Periods')
                axes[1,1].set_xlabel('Time Period')
                axes[1,1].set_ylabel('Charging Count')
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Strategy analysis chart saved to: {save_path}")
        
        return fig


