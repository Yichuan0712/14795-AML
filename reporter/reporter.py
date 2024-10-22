def get_nth_money_laundering(df, y, n):
    if len(y) != len(df):
        raise ValueError("The length of y must be equal to the number of rows in df.")

    df = df.drop(columns=['Is_laundering'])

    filtered_df = df[y == 1]

    if n < 0 or n >= len(filtered_df):
        raise IndexError(f"Index {n} is out of bounds for the filtered DataFrame.")

    return filtered_df.iloc[n]


def clean_data(input_df):
    alert_message = str({k: (True if v == 1 else v) for k, v in dict(input_df).items() if v != 0})
    return alert_message
