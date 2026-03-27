import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from module6_risk_detection import get_food_nutrients, rule_engine, train_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar as cal
from module4_nlp_preprocessing import (
    extract_entities_with_metadata,
    extract_relations_from_data,
    extract_triples_from_data,
    build_graph_dynamic,
    visualize_graph_dynamic,
)
API_BASE = "http://127.0.0.1:5000"



def show_kg_page():

    st.title("Knowledge Graph Explorer")

    selected_dataset = st.session_state.get("selected_dataset")

    if not selected_dataset:
        st.warning("Please select a dataset from Data Sources first.")
        return

    dataset_source = selected_dataset.get("source", "unknown")
    dataset_label = (
        selected_dataset.get("query")
        or selected_dataset.get("entity")
        or selected_dataset.get("title")
        or selected_dataset.get("description")
        or "Dataset"
    )
    st.success(f"Dataset: {dataset_source} - {dataset_label}")

    entities = extract_entities_with_metadata(selected_dataset)
    relations = extract_relations_from_data(selected_dataset)
    triples = extract_triples_from_data(selected_dataset)

    if not entities:
        st.warning("No entities found.")
        return

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Entities", len(entities))
    metric_col2.metric("Relations", len(relations))
    metric_col3.metric("Triples", len(triples))

    graph_entities = [(item["text"], item["label"]) for item in entities]
    graph = build_graph_dynamic(graph_entities, data=selected_dataset)

    if len(graph.nodes) == 0:
        st.warning("Graph could not be generated.")
        return

    st.markdown("### Knowledge Graph")
    html_file = visualize_graph_dynamic(graph)

    import streamlit.components.v1 as components
    with open(html_file, "r", encoding="utf-8") as f:
        components.html(f.read(), height=650)

    with st.expander("View extracted entities"):
        st.dataframe(pd.DataFrame(entities), use_container_width=True, hide_index=True)

    with st.expander("View extracted relations"):
        if relations:
            st.dataframe(pd.DataFrame(relations), use_container_width=True, hide_index=True)
        else:
            st.info("No relationships found.")

    with st.expander("View extracted triples"):
        if triples:
            st.dataframe(pd.DataFrame(triples), use_container_width=True, hide_index=True)
        else:
            st.info("No triples generated.")


def show_nutrient_page():
    if "show_graph" not in st.session_state:
        st.session_state.show_graph = False

    st.markdown("## Nutrient Analysis")
    st.markdown("Enter any food item to view its detailed nutrient breakdown from the USDA database.")

    st.markdown("---")

    food = st.text_input("Food", placeholder="e.g. Spinach, Salmon, Rice, Mango...")

    if st.button("Show Nutrient Details"):
        if not food.strip():
            st.warning("Please enter a food item.")
        else:
            st.session_state.show_graph = True
            st.session_state.analysis_food = food.strip()

    if st.session_state.show_graph:
        a_food = st.session_state.get("analysis_food", "")

        with st.spinner("Fetching nutrients..."):
            nutrients = get_food_nutrients(a_food)

        if nutrients is None:
            st.error(f"Could not fetch nutrient data for '{a_food}' from USDA database.")
        else:
            st.markdown("### Nutrient Breakdown")
            ncol1, ncol2, ncol3, ncol4 = st.columns(4)
            with ncol1:
                st.metric("Carbs", f"{nutrients['carbs']}g")
            with ncol2:
                st.metric("Sugar", f"{nutrients['sugar']}g")
            with ncol3:
                st.metric("Protein", f"{nutrients['protein']}g")
            with ncol4:
                st.metric("Fat", f"{nutrients['fat']}g")

            st.markdown("---")
            
            fig = go.Figure(data=[go.Bar(
                x=["Carbs", "Sugar", "Protein", "Fat"],
                y=[nutrients["carbs"], nutrients["sugar"], nutrients["protein"], nutrients["fat"]],
                marker_color=["#6366f1", "#f59e0b", "#10b981", "#ef4444"]
            )])
            fig.update_layout(
                title=f"Nutrient Composition - {a_food}",
                yaxis_title="Grams (g)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)



def show_dashboard():

    st.markdown("""
    <style>

    .stApp{
    background:#ffffff;
    color:#1a1a2e;
    font-family: 'Segoe UI', sans-serif;
    }

    .block-container{
    padding-left:40px !important;
    padding-right:40px !important;
    }

    h1{color:#0f4fa8;}
    h2{color:#0f8fd4;}
    h3{color:#0f518f;}
    h4{color:#11b4a6;}

    [data-baseweb="select"]{
    background:#f0f4ff;
    border-radius:10px;
    border:1px solid #c7d7f0;
    }

    [data-testid="metric-container"]{
    background:linear-gradient(135deg,#e0f0ff,#d0faf4);
    padding:16px;
    border-radius:14px;
    box-shadow:0px 4px 14px rgba(15,79,168,0.12);
    color:#0f4fa8;
    }

    img{
    border-radius:12px;
    box-shadow:0px 4px 14px rgba(15,79,168,0.12);
    }

    </style>
    """, unsafe_allow_html=True)

    user = st.session_state.get("user", {})
    name = user.get("full_name", "User")

    profile = st.session_state.get("loaded_profile", {}) or {}
    weight = profile.get("weight_kg", 0)
    sleep = profile.get("sleep_hours", 6.5)
    water = profile.get("water_intake", 1.5)

    profile_pic = st.session_state.get("profile_pic", None)
    if profile_pic is not None:
        st.sidebar.image(profile_pic, width=300)
    else:
        st.sidebar.image(
        "https://images.unsplash.com/photo-1576091160550-2173dba999ef",
        width=300
        )

    st.sidebar.markdown("### User Profile")
    st.sidebar.write(f"Name: {name}")
    st.sidebar.write("Goal: Healthy Diet")

    st.sidebar.progress(70)

    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
    "Navigation",
    [
    "Dashboard",
    "Medicine & Food Safety",
    "Knowledge Graph",
    "Meal Plan",
    "Feedback"
    ]
    )

    food_list=[
    "Spinach","Broccoli","Carrot","Cucumber","Rice","Apple",
    "Banana","Milk","Bread","Chicken","Fish","Egg","Yogurt",
    "Beans","Oats","Potato","Tomato","Avocado","Orange","Almonds"
    ]

    medicine_list=[
    "Warfarin","Aspirin","Metformin","Ibuprofen","Paracetamol",
    "Amoxicillin","Atorvastatin"
    ]

    food_interactions={
    "Warfarin":{
    "avoid":["Spinach","Broccoli","Kale"],
    "safe":["Carrot","Cucumber","Rice"]
    },
    "Aspirin":{
    "avoid":["Fish","Garlic"],
    "safe":["Apple","Oats","Banana"]
    },
    "Metformin":{
    "avoid":["Sugary Foods"],
    "safe":["Beans","Oats","Vegetables"]
    }
    }

    st.markdown("## Health Dashboard")

    # Show selected dataset if any (from data_sources_page)

    selected_dataset = st.session_state.get("selected_dataset")
    if selected_dataset:
        dataset_source = selected_dataset.get("source", "unknown")
        dataset_label = (
            selected_dataset.get("query")
            or selected_dataset.get("entity")
            or selected_dataset.get("title")
            or selected_dataset.get("description")
            or "Dataset"
        )
        st.success(f"Selected Dataset: {dataset_source} - {dataset_label}")

    if menu=="Dashboard":

        col1,col2=st.columns([4,1])

        with col1:
            st.markdown(f"## Welcome {name}")
            st.markdown("### MedGraphX Smart Health Dashboard")

        with col2:
            now=datetime.now()
            st.markdown(f"<p style='font-size:24px; font-weight:700; color:#0f4fa8;'>Date: {now.strftime('%d %B %Y')}</p>", unsafe_allow_html=True)

        st.markdown("---")

        col1,col2,col3=st.columns(3)

        with col1:
            st.metric("Weight", f"{weight} kg")

        with col2:
            st.metric("Sleep", f"{sleep} hrs")

        with col3:
            st.metric("Water Intake", f"{water} L")

        st.markdown("---")

        health_data=pd.DataFrame({
            "Week":["Week 1","Week 2","Week 3","Week 4"],
            "Weight":[80,79.5,78.7,78],
            "Steps":[7200,7600,7900,8050],
            "Sleep":[5.8,6.1,6.3,6.5],
            "Water":[0.9,1.1,1.2,1.3]
        })

        st.subheader("Weekly Health Insights")

        

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Weight (kg)", "Steps", "Sleep (hrs)", "Water (L)"),
            vertical_spacing=0.18,
            horizontal_spacing=0.12
        )

        fig.add_trace(go.Scatter(
            x=health_data["Week"], y=health_data["Weight"],
            mode="lines+markers", name="Weight",
            line=dict(color="#6366f1", width=3), marker=dict(size=9)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=health_data["Week"], y=health_data["Steps"],
            mode="lines+markers", name="Steps",
            line=dict(color="#f59e0b", width=3), marker=dict(size=9)
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=health_data["Week"], y=health_data["Sleep"],
            mode="lines+markers", name="Sleep",
            line=dict(color="#ef4444", width=3), marker=dict(size=9)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=health_data["Week"], y=health_data["Water"],
            mode="lines+markers", name="Water",
            line=dict(color="#10b981", width=3), marker=dict(size=9)
        ), row=2, col=2)

        fig.update_layout(
            height=500,
            template="plotly_white",
            showlegend=False,
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Understanding the Health Trends")

        st.write("**Steps Trend** - Displays physical activity levels through daily steps.")
        st.write("**Sleep Trend** - Represents average sleep hours each week.")
        st.write("**Water Intake** - Indicates hydration levels over time.")
        st.write("**Weight Trend** - Shows how your body weight changes weekly.")

        st.info("Tip: Increasing steps and water intake while maintaining good sleep helps manage weight effectively.")

    elif menu=="Medicine & Food Safety":

        st.markdown("# Medicine & Food Safety")

        medicine=st.selectbox("Select Medicine",medicine_list)
        food=st.selectbox("Select Food",food_list)

        st.markdown("---")

        avoid=[]
        safe=[]

        if medicine in food_interactions:
            avoid=food_interactions[medicine]["avoid"]
            safe=food_interactions[medicine]["safe"]

        col1,col2=st.columns(2)

        with col1:
            st.subheader("Foods To Avoid")
            for f in avoid:
                st.write(f"- {f}")

        with col2:
            st.subheader("Safe Foods")
            for f in safe:
                st.write(f"- {f}")

        st.markdown("---")

        st.subheader("Medicine Tips")

        st.write("- Always take medicines with water")
        st.write("- Avoid alcohol while taking medicines")
        st.write("- Follow doctor prescriptions")

    elif menu=="Knowledge Graph":
        show_kg_page()

    elif menu=="Meal Plan":

        

        st.markdown("# 🍽️ Weekly Meal Planner")

        # Determine diet preference
        diet_pref = profile.get("diet_preference", "Vegetarian") or "Vegetarian"
        is_veg = diet_pref in ["Vegetarian", "Vegan"]

        # Non-veg meal plan
        nonveg_meal_plan = {
            "Monday": {
                "breakfast": {"meal": "Oatmeal with Fruits", "details": "Whole grain oats topped with banana, berries, and honey.", "calories": "350 kcal", "img": "https://images.unsplash.com/photo-1517673400267-0251440c45dc?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Quinoa Salad", "details": "Quinoa with roasted vegetables, chickpeas, and lemon dressing.", "calories": "420 kcal", "img": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Grilled Chicken & Rice", "details": "Herb-grilled chicken breast with brown rice and steamed greens.", "calories": "520 kcal", "img": "https://images.unsplash.com/photo-1598515214211-89d3c73ae83b?w=300&h=200&fit=crop"},
            },
            "Tuesday": {
                "breakfast": {"meal": "Whole Wheat Pancakes", "details": "Fluffy pancakes with fresh berries and maple syrup.", "calories": "380 kcal", "img": "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Chicken Salad", "details": "Grilled chicken on mixed greens, cherry tomatoes, and vinaigrette.", "calories": "420 kcal", "img": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Pasta Primavera", "details": "Whole wheat pasta with sauteed seasonal vegetables in olive oil.", "calories": "480 kcal", "img": "https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=300&h=200&fit=crop"},
            },
            "Wednesday": {
                "breakfast": {"meal": "Scrambled Eggs & Toast", "details": "Fluffy scrambled eggs with whole grain toast and avocado.", "calories": "360 kcal", "img": "https://images.unsplash.com/photo-1525351484163-7529414344d8?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Vegetable Soup", "details": "Hearty soup with carrots, celery, potatoes, and spinach.", "calories": "280 kcal", "img": "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Lean Steak & Veggies", "details": "Grilled lean steak with roasted sweet potatoes and asparagus.", "calories": "550 kcal", "img": "https://images.unsplash.com/photo-1558030006-450675393462?w=300&h=200&fit=crop"},
            },
            "Thursday": {
                "breakfast": {"meal": "Smoothie Bowl", "details": "Blueberry-banana smoothie bowl with granola and chia seeds.", "calories": "320 kcal", "img": "https://images.unsplash.com/photo-1590301157890-4810ed352733?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Turkey Wrap", "details": "Whole wheat wrap with turkey, lettuce, tomato, and hummus.", "calories": "400 kcal", "img": "https://images.unsplash.com/photo-1626700051175-6818013e1d4f?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Grilled Fish", "details": "Lemon-herb grilled tilapia with steamed broccoli and quinoa.", "calories": "390 kcal", "img": "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=300&h=200&fit=crop"},
            },
            "Friday": {
                "breakfast": {"meal": "Fruit Bowl", "details": "Seasonal fresh fruits with Greek yogurt and granola topping.", "calories": "300 kcal", "img": "https://images.unsplash.com/photo-1490474418585-ba9bad8fd0ea?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Falafel Pita", "details": "Crispy falafel in whole wheat pita with tahini and veggies.", "calories": "450 kcal", "img": "https://images.unsplash.com/photo-1558315894-8021bea13aa3?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Butter Chicken", "details": "Creamy butter chicken with naan bread and basmati rice.", "calories": "530 kcal", "img": "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?w=300&h=200&fit=crop"},
            },
            "Saturday": {
                "breakfast": {"meal": "Avocado Toast", "details": "Whole wheat toast with mashed avocado, poached egg, and chili flakes.", "calories": "380 kcal", "img": "https://images.unsplash.com/photo-1541519227354-08fa5d50c44d?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Sushi Bowl", "details": "Deconstructed sushi with salmon, rice, avocado, and edamame.", "calories": "440 kcal", "img": "https://images.unsplash.com/photo-1579871494447-9811cf80d66c?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Veggie Flatbread", "details": "Whole wheat flatbread with grilled vegetables, mozzarella, and pesto.", "calories": "460 kcal", "img": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=300&h=200&fit=crop"},
            },
            "Sunday": {
                "breakfast": {"meal": "Waffles & Berries", "details": "Whole grain waffles with mixed berries and a light cream.", "calories": "370 kcal", "img": "https://images.unsplash.com/photo-1562376552-0d160a2f238d?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Lentil Stew", "details": "Hearty lentil and vegetable stew with crusty bread.", "calories": "410 kcal", "img": "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Salmon with Vegetables", "details": "Oven-baked salmon fillet with roasted asparagus and sweet potato.", "calories": "450 kcal", "img": "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=300&h=200&fit=crop"},
            },
        }

        # Vegetarian meal plan
        veg_meal_plan = {
            "Monday": {
                "breakfast": {"meal": "Oatmeal with Fruits", "details": "Whole grain oats topped with banana, berries, and honey.", "calories": "350 kcal", "img": "https://images.unsplash.com/photo-1517673400267-0251440c45dc?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Quinoa Salad", "details": "Quinoa with roasted vegetables, chickpeas, and lemon dressing.", "calories": "420 kcal", "img": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Paneer Tikka Masala", "details": "Creamy tomato-based curry with paneer and aromatic spices.", "calories": "480 kcal", "img": "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?w=300&h=200&fit=crop"},
            },
            "Tuesday": {
                "breakfast": {"meal": "Whole Wheat Pancakes", "details": "Fluffy pancakes with fresh berries and maple syrup.", "calories": "380 kcal", "img": "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Greek Salad", "details": "Cucumber, tomatoes, olives, feta cheese with olive oil dressing.", "calories": "350 kcal", "img": "https://images.unsplash.com/photo-1540420773420-3366772f4999?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Pasta Primavera", "details": "Whole wheat pasta with sauteed seasonal vegetables in olive oil.", "calories": "480 kcal", "img": "https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=300&h=200&fit=crop"},
            },
            "Wednesday": {
                "breakfast": {"meal": "Avocado Toast", "details": "Whole wheat toast with mashed avocado, tomato, and seeds.", "calories": "340 kcal", "img": "https://images.unsplash.com/photo-1541519227354-08fa5d50c44d?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Vegetable Soup", "details": "Hearty soup with carrots, celery, potatoes, and spinach.", "calories": "280 kcal", "img": "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Veggie Tacos", "details": "Soft corn tortillas with black beans, peppers, salsa, and guacamole.", "calories": "420 kcal", "img": "https://images.unsplash.com/photo-1551504734-5ee1c4a1479b?w=300&h=200&fit=crop"},
            },
            "Thursday": {
                "breakfast": {"meal": "Smoothie Bowl", "details": "Blueberry-banana smoothie bowl with granola and chia seeds.", "calories": "320 kcal", "img": "https://images.unsplash.com/photo-1590301157890-4810ed352733?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Falafel Pita", "details": "Crispy falafel in whole wheat pita with tahini and veggies.", "calories": "450 kcal", "img": "https://images.unsplash.com/photo-1558315894-8021bea13aa3?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Dal & Rice", "details": "Yellow lentil dal with cumin tempering and steamed basmati rice.", "calories": "400 kcal", "img": "https://images.unsplash.com/photo-1596797038530-2c107229654b?w=300&h=200&fit=crop"},
            },
            "Friday": {
                "breakfast": {"meal": "Fruit Bowl", "details": "Seasonal fresh fruits with Greek yogurt and granola topping.", "calories": "300 kcal", "img": "https://images.unsplash.com/photo-1490474418585-ba9bad8fd0ea?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Chana Masala", "details": "Spiced chickpea curry with tomatoes, onions, and fresh herbs.", "calories": "430 kcal", "img": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Vegetable Curry", "details": "Coconut-based curry with mixed vegetables and basmati rice.", "calories": "470 kcal", "img": "https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?w=300&h=200&fit=crop"},
            },
            "Saturday": {
                "breakfast": {"meal": "Veggie Omelette", "details": "Egg-free chickpea omelette with bell peppers, spinach, and mushrooms.", "calories": "360 kcal", "img": "https://images.unsplash.com/photo-1525351484163-7529414344d8?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Buddha Bowl", "details": "Brown rice, roasted sweet potato, avocado, edamame, and tahini.", "calories": "440 kcal", "img": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Veggie Flatbread", "details": "Whole wheat flatbread with grilled vegetables, mozzarella, and pesto.", "calories": "460 kcal", "img": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=300&h=200&fit=crop"},
            },
            "Sunday": {
                "breakfast": {"meal": "Waffles & Berries", "details": "Whole grain waffles with mixed berries and a light cream.", "calories": "370 kcal", "img": "https://images.unsplash.com/photo-1562376552-0d160a2f238d?w=300&h=200&fit=crop"},
                "lunch": {"meal": "Lentil Stew", "details": "Hearty lentil and vegetable stew with crusty bread.", "calories": "410 kcal", "img": "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=300&h=200&fit=crop"},
                "dinner": {"meal": "Mushroom Risotto", "details": "Creamy arborio rice with sauteed mushrooms, parmesan, and herbs.", "calories": "460 kcal", "img": "https://images.unsplash.com/photo-1476124369491-e7addf5db371?w=300&h=200&fit=crop"},
            },
        }

        # Select meal plan based on preference
        meal_plan = veg_meal_plan if is_veg else nonveg_meal_plan

        # Show diet badge
        badge_color = "#2e7d32" if is_veg else "#c62828"
        badge_label = "Vegetarian" if is_veg else "Non-Vegetarian"
        st.markdown(f"<span style='background:{badge_color};color:white;padding:5px 16px;border-radius:20px;font-size:14px;font-weight:600;'>{badge_label}</span>", unsafe_allow_html=True)
        st.markdown("")

        # Session state for selected day and month navigation
        if "meal_selected_day" not in st.session_state:
            st.session_state.meal_selected_day = None
        if "meal_cal_year" not in st.session_state:
            st.session_state.meal_cal_year = datetime.now().year
        if "meal_cal_month" not in st.session_state:
            st.session_state.meal_cal_month = datetime.now().month

        year = st.session_state.meal_cal_year
        month = st.session_state.meal_cal_month
        today = datetime.now()

        # Month navigation
        month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]

        # Calendar CSS - compact size
        st.markdown("""
        <style>
        .meal-cal-wrap div[data-testid="stHorizontalBlock"] .stButton > button {
            height: 48px !important;
            min-height: 48px !important;
            border-radius: 8px !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            padding: 2px 4px !important;
            transition: all 0.2s ease !important;
        }
        .cal-header {
            background: linear-gradient(135deg, #0f4fa8, #0f8fd4);
            color: white;
            text-align: center;
            padding: 6px 0;
            border-radius: 8px;
            font-weight: 700;
            font-size: 12px;
            margin-bottom: 2px;
        }
        .meal-card {
            background: linear-gradient(135deg, #e8f4fd, #d0faf4);
            border-radius: 16px;
            padding: 20px 24px;
            margin: 10px 0;
            box-shadow: 0 4px 16px rgba(15,79,168,0.12);
            border-left: 5px solid #0f4fa8;
        }
        .meal-card h3 { color: #0f4fa8; margin-top: 0; }
        .meal-section {
            background: white;
            border-radius: 12px;
            padding: 14px 18px;
            margin: 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 4px solid #0f8fd4;
        }
        .meal-section h4 { margin: 0 0 6px 0; color: #0f4fa8; font-size: 15px; }
        .meal-section p { margin: 4px 0; color: #444; font-size: 14px; }
        .meal-section-inner { display: flex; gap: 16px; align-items: flex-start; }
        .meal-section-img { width: 120px; height: 80px; border-radius: 10px; object-fit: cover; flex-shrink: 0; }
        .meal-section-text { flex: 1; }
        .meal-badge {
            background: #0f4fa8;
            color: white;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
            margin-top: 6px;
        }
        </style>
        """, unsafe_allow_html=True)

        nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
        with nav_col1:
            if st.button("< Prev", key="meal_prev_month", use_container_width=True):
                if month == 1:
                    st.session_state.meal_cal_month = 12
                    st.session_state.meal_cal_year = year - 1
                else:
                    st.session_state.meal_cal_month = month - 1
                st.session_state.meal_selected_day = None
                st.rerun()
        with nav_col2:
            st.markdown(f"<h3 style='text-align:center;margin:0;color:#0f4fa8;'>{month_names[month-1]} {year}</h3>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("Next >", key="meal_next_month", use_container_width=True):
                if month == 12:
                    st.session_state.meal_cal_month = 1
                    st.session_state.meal_cal_year = year + 1
                else:
                    st.session_state.meal_cal_month = month + 1
                st.session_state.meal_selected_day = None
                st.rerun()

        # Constrain calendar width
        _, cal_col, _ = st.columns([1, 4, 1])
        with cal_col:
            st.markdown("<div class='meal-cal-wrap'>", unsafe_allow_html=True)

            # Weekday headers
            day_headers = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            header_cols = st.columns(7)
            for i, dh in enumerate(day_headers):
                with header_cols[i]:
                    st.markdown(f"<div class='cal-header'>{dh}</div>", unsafe_allow_html=True)

            # Build calendar grid
            month_cal = cal.monthcalendar(year, month)
            weekday_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

            for week in month_cal:
                cols = st.columns(7)
                for i, day in enumerate(week):
                    with cols[i]:
                        if day == 0:
                            st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
                        else:
                            is_today = (day == today.day and month == today.month and year == today.year)
                            is_selected = (st.session_state.meal_selected_day == day)

                            btn_label = f"{day}📍" if is_today else str(day)

                            if st.button(
                                btn_label,
                                key=f"meal_day_{year}_{month}_{day}",
                                use_container_width=True,
                                type="primary" if is_selected else "secondary"
                            ):
                                st.session_state.meal_selected_day = day
                                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # Show selected day meal details (breakfast, lunch, dinner)
        if st.session_state.meal_selected_day is not None:
            sel_day = st.session_state.meal_selected_day
            sel_date = datetime(year, month, sel_day)
            wday_name = weekday_names[sel_date.weekday()]
            day_meals = meal_plan[wday_name]

            st.markdown(f"""
            <div class="meal-card">
                <h3>{wday_name}, {month_names[month-1]} {sel_day}, {year}</h3>
                <div class="meal-section">
                    <h4>Breakfast</h4>
                    <div class="meal-section-inner">
                        <img class="meal-section-img" src="{day_meals['breakfast']['img']}" alt="Breakfast" />
                        <div class="meal-section-text">
                            <p><strong>{day_meals['breakfast']['meal']}</strong></p>
                            <p>{day_meals['breakfast']['details']}</p>
                            <span class="meal-badge">Calories: {day_meals['breakfast']['calories']}</span>
                        </div>
                    </div>
                </div>
                <div class="meal-section">
                    <h4>Lunch</h4>
                    <div class="meal-section-inner">
                        <img class="meal-section-img" src="{day_meals['lunch']['img']}" alt="Lunch" />
                        <div class="meal-section-text">
                            <p><strong>{day_meals['lunch']['meal']}</strong></p>
                            <p>{day_meals['lunch']['details']}</p>
                            <span class="meal-badge">Calories: {day_meals['lunch']['calories']}</span>
                        </div>
                    </div>
                </div>
                <div class="meal-section">
                    <h4>Dinner</h4>
                    <div class="meal-section-inner">
                        <img class="meal-section-img" src="{day_meals['dinner']['img']}" alt="Dinner" />
                        <div class="meal-section-text">
                            <p><strong>{day_meals['dinner']['meal']}</strong></p>
                            <p>{day_meals['dinner']['details']}</p>
                            <span class="meal-badge">Calories: {day_meals['dinner']['calories']}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("Healthy Eating Tips")

        st.write("- Drink 2-3 liters of water daily")
        st.write("- Maintain balanced diet")
        st.write("- Exercise regularly")
        st.write("- Avoid junk food")

        st.markdown("---")

        st.subheader("Healthy Snack Ideas")

        st.write("- Mixed nuts")
        st.write("- Yogurt with honey")
        st.write("- Whole grain sandwiches")

        st.markdown("---")

        st.subheader("Hydration Reminder")

        st.write("- Drink water every 2-3 hours")
        st.write("- Carry a reusable water bottle")
        st.write("- Increase water intake during exercise")

        st.markdown("---")

        st.subheader("Healthy Lifestyle Tips")

        st.write("- Walk at least 8000 steps daily")
        st.write("- Sleep 7-8 hours every night")
        st.write("- Reduce sugar intake")
        st.write("- Stay physically active")


    elif menu=="Feedback":

        st.markdown("# Feedback")
        st.markdown("Share your experience with MedGraphX so we can keep improving the dashboard for you.")

        token = st.session_state.get("token")
        if not token:
            st.warning("Please login again to submit feedback.")
            return

        rating_options = [1, 2, 3, 4, 5]
        rating_help = "1 = needs improvement, 5 = excellent"

        with st.form("user_feedback_form"):
            risk_analysis = st.select_slider("Risk Analysis", options=rating_options, value=4, help=rating_help)
            nutrient_analysis = st.select_slider("Nutrient Analysis", options=rating_options, value=4, help=rating_help)
            knowledge_mapping = st.select_slider("Knowledge Mapping", options=rating_options, value=4, help=rating_help)
            data_extraction = st.select_slider("Data Extraction", options=rating_options, value=4, help=rating_help)
            meal_planning = st.select_slider("Meal Planning", options=rating_options, value=4, help=rating_help)
            comments = st.text_area(
                "Comments",
                placeholder="Tell us what is working well and what you would like us to improve.",
                height=160,
            )

            submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

        if submitted:
            payload = {
                "risk_analysis": risk_analysis,
                "nutrient_analysis": nutrient_analysis,
                "knowledge_mapping": knowledge_mapping,
                "data_extraction": data_extraction,
                "meal_planning": meal_planning,
                "comments": comments.strip(),
            }

            try:
                response = requests.post(
                    f"{API_BASE}/feedback",
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10,
                )

                if response.status_code == 201:
                    st.success("Thank you. Your feedback was submitted successfully.")
                else:
                    error_message = "Failed to submit feedback."
                    try:
                        body = response.json()
                        error_message = body.get("error") or body.get("message") or error_message
                    except Exception:
                        pass
                    st.error(error_message)
            except Exception as exc:
                st.error(f"Could not submit feedback right now: {exc}")




