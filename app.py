# Old (matplotlib barh)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(bank_kpis_sorted["bank_name"], bank_kpis_sorted["customers"])
st.pyplot(fig)

# New (Altair)
chart = alt.Chart(bank_kpis).mark_bar().encode(
    x="customers:Q",
    y=alt.Y("bank_name:N", sort="-x"),
    color="bank_name:N",
    tooltip=["bank_name", "customers"]
).properties(title="Customer Distribution by Bank", width=600)
st.altair_chart(chart, use_container_width=True)
