#include <vector>
#include <stdexcept>
#include <cmath>

class StandardScaler {
public:
    std::vector<double> mean_, std_;

    // X is a 2D matrix: rows = samples, cols = features
    StandardScaler& fit(const std::vector<std::vector<double>>& X) {
        if (X.empty()) throw std::invalid_argument("Input matrix is empty");

        size_t nRows = X.size();
        size_t nCols = X[0].size();

        mean_.assign(nCols, 0.0);
        std_.assign(nCols, 0.0);

        // Compute column-wise mean
        for (size_t col = 0; col < nCols; ++col) {
            double sum = 0.0;
            for (size_t row = 0; row < nRows; ++row)
                sum += X[row][col];
            mean_[col] = sum / nRows;
        }

        // Compute column-wise std (population std, ddof=0, matching numpy default)
        for (size_t col = 0; col < nCols; ++col) {
            double variance = 0.0;
            for (size_t row = 0; row < nRows; ++row) {
                double diff = X[row][col] - mean_[col];
                variance += diff * diff;
            }
            std_[col] = std::sqrt(variance / nRows);
        }

        return *this;
    }

    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>& X) const {
        if (mean_.empty() || std_.empty())
            throw std::runtime_error("Scaler has not been fitted yet");

        size_t nRows = X.size();
        size_t nCols = X[0].size();

        std::vector<std::vector<double>> X_scaled(nRows, std::vector<double>(nCols, 0.0));

        for (size_t row = 0; row < nRows; ++row)
            for (size_t col = 0; col < nCols; ++col)
                X_scaled[row][col] = (std_[col] > 0)
                    ? (X[row][col] - mean_[col]) / std_[col]
                    : 0.0;

        return X_scaled;
    }

    // Convenience: fit then transform in one call
    std::vector<std::vector<double>> fit_transform(const std::vector<std::vector<double>>& X) {
        return fit(X).transform(X);
    }
};
