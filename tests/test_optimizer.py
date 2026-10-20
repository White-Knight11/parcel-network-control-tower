import unittest

import numpy as np
import pandas as pd

from app import (
    build_route_lines,
    generate_data,
    haversine_distance_miles,
    optimize_routes,
)


class RoutingTests(unittest.TestCase):
    def test_generated_network_is_new_jersey_focused_and_includes_jersey_city(self):
        orders, facilities = generate_data(200)

        self.assertIn("Jersey_City_Hub", facilities["Facility_ID"].tolist())
        self.assertIn("Jersey City", orders["Customer_City"].values)
        self.assertLessEqual(
            set(orders["Customer_City"]),
            {
                "Jersey City",
                "Newark",
                "Edison",
                "Trenton",
                "Atlantic City",
                "Mahwah",
                "Toms River",
                "Camden",
            },
        )
        self.assertTrue(orders["Customer_Lat"].between(39.0, 41.35).all())
        self.assertTrue(orders["Customer_Lon"].between(-75.6, -73.85).all())

    def test_route_lines_connect_facility_to_assigned_order(self):
        orders = pd.DataFrame(
            {
                "Order_ID": [1],
                "Customer_Lat": [40.1],
                "Customer_Lon": [-74.1],
                "Assigned_Node": ["Jersey_City_Hub"],
            }
        )
        facilities = pd.DataFrame(
            {
                "Facility_ID": ["Jersey_City_Hub"],
                "Lat": [40.7],
                "Lon": [-74.0],
            }
        )

        lines = build_route_lines(orders, facilities, route_limit=1)

        self.assertEqual(lines.iloc[0]["Lat"], 40.7)
        self.assertEqual(lines.iloc[0]["Lon"], -74.0)
        self.assertEqual(lines.iloc[1]["Lat"], 40.1)
        self.assertEqual(lines.iloc[1]["Lon"], -74.1)
        self.assertTrue(pd.isna(lines.iloc[2]["Lat"]))

    def test_haversine_distance_is_close_to_known_distance(self):
        distance = haversine_distance_miles(
            [40.7128], [-74.0060], [42.3601], [-71.0589]
        )[0, 0]
        self.assertAlmostEqual(distance, 190, delta=5)

    def test_capacity_is_respected_and_excess_orders_are_unfulfilled(self):
        orders = pd.DataFrame(
            {
                "Order_ID": [1, 2, 3],
                "Customer_Lat": [40.0, 40.1, 40.2],
                "Customer_Lon": [-74.0, -74.1, -74.2],
                "Gross_Margin_$": [20.0, 20.0, 20.0],
            }
        )
        facilities = pd.DataFrame(
            {
                "Facility_ID": ["Warehouse", "Store"],
                "Lat": [40.0, 40.2],
                "Lon": [-74.0, -74.2],
                "Handling_Cost": [1.0, 2.0],
                "Daily_Capacity": [1, 1],
            }
        )

        result = optimize_routes(
            orders,
            facilities,
            freight_rate=0.1,
            packaging_cost=0.5,
            unfulfilled_penalty=100,
        )

        assigned = result[result["Assigned_Node"] != "Unfulfilled"]
        counts = assigned["Assigned_Node"].value_counts()
        self.assertEqual(len(assigned), 2)
        self.assertLessEqual(counts.get("Warehouse", 0), 1)
        self.assertLessEqual(counts.get("Store", 0), 1)
        self.assertEqual((result["Assigned_Node"] == "Unfulfilled").sum(), 1)
        self.assertTrue(result.loc[result["Assigned_Node"] == "Unfulfilled", "Cost_to_Serve"].isna().all())

    def test_zero_capacity_leaves_all_orders_unfulfilled(self):
        orders = pd.DataFrame(
            {
                "Order_ID": [1],
                "Customer_Lat": [40.0],
                "Customer_Lon": [-74.0],
                "Gross_Margin_$": [20.0],
            }
        )
        facilities = pd.DataFrame(
            {
                "Facility_ID": ["Warehouse"],
                "Lat": [40.0],
                "Lon": [-74.0],
                "Handling_Cost": [1.0],
                "Daily_Capacity": [0],
            }
        )

        result = optimize_routes(
            orders,
            facilities,
            freight_rate=0.1,
            packaging_cost=0.5,
            unfulfilled_penalty=100,
        )

        self.assertEqual(result.loc[0, "Assigned_Node"], "Unfulfilled")
        self.assertTrue(np.isnan(result.loc[0, "Cost_to_Serve"]))


if __name__ == "__main__":
    unittest.main()
