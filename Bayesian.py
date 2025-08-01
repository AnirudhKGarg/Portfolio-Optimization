import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class BayesianSharpeRatio:
    """
    Bayesian estimation of Sharpe ratio with conjugate priors.
    
    Uses Normal-Inverse-Gamma conjugate prior for (mu, sigma^2) parameters
    of excess returns, allowing analytical posterior updates.
    """
    
    def __init__(self, 
                 mu_prior: float = 0.0,
                 kappa_prior: float = 1.0,
                 alpha_prior: float = 1.0,
                 beta_prior: float = 1.0):
        """
        Initialize with Normal-Inverse-Gamma prior parameters.
        
        Prior: mu | sigma^2 ~ N(mu_prior, sigma^2/kappa_prior)
               sigma^2 ~ InvGamma(alpha_prior, beta_prior)
        
        Args:
            mu_prior: Prior mean for excess return
            kappa_prior: Prior precision parameter (higher = more confident)
            alpha_prior: Prior shape parameter for variance
            beta_prior: Prior scale parameter for variance
        """
        self.mu_prior = mu_prior
        self.kappa_prior = kappa_prior
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Initialize posterior parameters
        self.reset_posterior()
    
    def reset_posterior(self):
        """Reset posterior to prior values."""
        self.mu_post = self.mu_prior
        self.kappa_post = self.kappa_prior
        self.alpha_post = self.alpha_prior
        self.beta_post = self.beta_prior
        self.n_obs = 0
    
    def update(self, excess_returns: np.ndarray):
        """
        Update posterior with new excess returns data.
        
        Args:
            excess_returns: Array of excess returns (returns - risk_free_rate)
        """
        n = len(excess_returns)
        sample_mean = np.mean(excess_returns)
        sample_var = np.var(excess_returns, ddof=1) if n > 1 else 0
        
        # Conjugate prior updates
        self.kappa_post = self.kappa_prior + n
        self.alpha_post = self.alpha_prior + n/2
        
        self.mu_post = (self.kappa_prior * self.mu_prior + n * sample_mean) / self.kappa_post
        
        self.beta_post = (self.beta_prior + 
                         0.5 * (n-1) * sample_var + 
                         0.5 * (self.kappa_prior * n * (sample_mean - self.mu_prior)**2) / self.kappa_post)
        
        self.n_obs += n
    
    def posterior_samples(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples from posterior distribution of (mu, sigma).
        
        Returns:
            Tuple of (mu_samples, sigma_samples)
        """
        # Sample sigma^2 from Inverse-Gamma
        sigma2_samples = stats.invgamma.rvs(self.alpha_post, scale=self.beta_post, size=n_samples)
        
        # Sample mu from Normal given sigma^2
        mu_samples = np.array([
            stats.norm.rvs(self.mu_post, np.sqrt(s2/self.kappa_post)) 
            for s2 in sigma2_samples
        ])
        
        sigma_samples = np.sqrt(sigma2_samples)
        
        return mu_samples, sigma_samples
    
    def sharpe_ratio_distribution(self, n_samples: int = 10000) -> np.ndarray:
        """
        Generate samples from posterior distribution of Sharpe ratio.
        
        Returns:
            Array of Sharpe ratio samples
        """
        mu_samples, sigma_samples = self.posterior_samples(n_samples)
        
        # Handle division by zero
        sharpe_samples = np.divide(mu_samples, sigma_samples, 
                                 out=np.zeros_like(mu_samples), 
                                 where=sigma_samples!=0)
        
        return sharpe_samples
    
    def sharpe_statistics(self, n_samples: int = 10000) -> dict:
        """
        Compute summary statistics for Sharpe ratio posterior.
        
        Returns:
            Dictionary with mean, std, quantiles, and probability metrics
        """
        sharpe_samples = self.sharpe_ratio_distribution(n_samples)
        
        return {
            'mean': np.mean(sharpe_samples),
            'std': np.std(sharpe_samples),
            'median': np.median(sharpe_samples),
            'q025': np.percentile(sharpe_samples, 2.5),
            'q975': np.percentile(sharpe_samples, 97.5),
            'prob_positive': np.mean(sharpe_samples > 0),
            'prob_greater_than_1': np.mean(sharpe_samples > 1.0),
            'samples': sharpe_samples
        }
    
    def classical_sharpe(self, excess_returns: np.ndarray) -> float:
        """Compute classical sample Sharpe ratio for comparison."""
        if len(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) if np.std(excess_returns, ddof=1) > 0 else 0.0
    
    def plot_posterior(self, n_samples: int = 10000, figsize: Tuple[int, int] = (12, 4)):
        """Plot posterior distributions of mu, sigma, and Sharpe ratio."""
        mu_samples, sigma_samples = self.posterior_samples(n_samples)
        sharpe_samples = self.sharpe_ratio_distribution(n_samples)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot mu posterior
        axes[0].hist(mu_samples, bins=50, alpha=0.7, density=True)
        axes[0].axvline(self.mu_post, color='red', linestyle='--', label=f'Posterior mean: {self.mu_post:.4f}')
        axes[0].set_xlabel('Excess Return (μ)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Posterior: Excess Return')
        axes[0].legend()
        
        # Plot sigma posterior
        axes[1].hist(sigma_samples, bins=50, alpha=0.7, density=True)
        sigma_mean = np.sqrt(self.beta_post / (self.alpha_post - 1)) if self.alpha_post > 1 else np.nan
        axes[1].axvline(sigma_mean, color='red', linestyle='--', label=f'Posterior mean: {sigma_mean:.4f}')
        axes[1].set_xlabel('Volatility (σ)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Posterior: Volatility')
        axes[1].legend()
        
        # Plot Sharpe ratio posterior
        axes[2].hist(sharpe_samples, bins=50, alpha=0.7, density=True)
        sharpe_mean = np.mean(sharpe_samples)
        axes[2].axvline(sharpe_mean, color='red', linestyle='--', label=f'Posterior mean: {sharpe_mean:.4f}')
        axes[2].axvline(0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_xlabel('Sharpe Ratio')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Posterior: Sharpe Ratio')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()


def example_usage():
    """Example usage of BayesianSharpeRatio class."""
    
    # Generate some sample data
    np.random.seed(42)
    true_mu = 0.08  # 8% annual excess return
    true_sigma = 0.15  # 15% annual volatility
    n_periods = 252  # Daily data for 1 year
    
    excess_returns = np.random.normal(true_mu/252, true_sigma/np.sqrt(252), n_periods)
    
    # Initialize Bayesian Sharpe ratio with weakly informative priors
    bayes_sr = BayesianSharpeRatio(
        mu_prior=0.0,      # Neutral prior on excess return
        kappa_prior=1.0,   # Low confidence in prior
        alpha_prior=2.0,   # Weak prior on variance
        beta_prior=0.01    # Weak prior on variance
    )
    
    # Update with data
    bayes_sr.update(excess_returns)
    
    # Get Sharpe ratio statistics
    stats_dict = bayes_sr.sharpe_statistics()
    classical_sr = bayes_sr.classical_sharpe(excess_returns)
    
    print("Bayesian Sharpe Ratio Analysis")
    print("=" * 40)
    print(f"Classical Sharpe Ratio: {classical_sr:.4f}")
    print(f"Bayesian Mean: {stats_dict['mean']:.4f}")
    print(f"Bayesian Std: {stats_dict['std']:.4f}")
    print(f"95% Credible Interval: [{stats_dict['q025']:.4f}, {stats_dict['q975']:.4f}]")
    print(f"P(Sharpe > 0): {stats_dict['prob_positive']:.3f}")
    print(f"P(Sharpe > 1): {stats_dict['prob_greater_than_1']:.3f}")
    print(f"Number of observations: {bayes_sr.n_obs}")
    
    # Plot results
    bayes_sr.plot_posterior()
    
    return bayes_sr, stats_dict

if __name__ == "__main__":
    example_usage()
