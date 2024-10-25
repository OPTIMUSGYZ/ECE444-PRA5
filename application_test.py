import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
import requests

from application import application


@pytest.fixture
def client():
    application.testing = True
    return application.test_client()


def test_empty_input(client):
    response = client.post('/predict', json={'text': ''})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_latency_and_performance():
    url = "http://ece444pra5-env.eba-3qdiupms.us-east-1.elasticbeanstalk.com/predict"
    test_inputs = [
        "BREAKING: Aliens land in New York City",
        "Scientists discover that the Earth is actually flat",
        "OnePlus officially shames One UI for its failure in system fluency.",
        "Apple Releases New AirPods Pro 2 Firmware Ahead of Hearing Aid Feature Launch."
    ]

    results = {text: [] for text in test_inputs}

    # Perform 100 API calls for each test case
    for i, text in enumerate(test_inputs, 1):
        csv_filename = f'latency_results_case_{i}.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f'{text}'])
            writer.writerow(['Request Number', 'Latency (seconds)'])

            for j in range(100):
                try:
                    start_time = time.time()
                    requests.post(url, json={'text': text}, timeout=10)
                    end_time = time.time()
                    latency = end_time - start_time
                    results[text].append(latency)

                    writer.writerow([j + 1, latency])
                except requests.exceptions.RequestException as e:
                    print(f"Request failed for Test Case {i}, Request {j + 1}: {e}")
                    writer.writerow([j + 1, "Failed"])

        # Generate boxplot for each test case
        plt.figure(figsize=(10, 6))
        plt.boxplot(results[text])
        plt.title(f'API Latency for Test Case {i}')
        plt.xlabel('Test Case')
        plt.ylabel('Latency (seconds)')
        plt.xticks([1], [f'Case {i}'])
        plt.savefig(f'latency_boxplot_case_{i}.png')
        plt.close()

    # Calculate and print average performance
    print("")
    for i, (text, latencies) in enumerate(results.items(), 1):
        if latencies:
            avg_latency = np.mean(latencies)
            print(f"Average latency for Test Case {i}: {avg_latency:.4f} seconds")
        else:
            print(f"No successful requests for Test Case {i}")


if __name__ == '__main__':
    pytest.main([__file__])
    print("\nRunning latency and performance test...")
    test_latency_and_performance()
