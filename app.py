import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

def predict_price(features):
    # Convert features to numpy array
    features_array = np.array([features['Area'], features['No. of Bedrooms'], features['Resale'],
                                features['MaintenanceStaff'], features['Gymnasium'], features['SwimmingPool'],
                                features['LandscapedGardens'], features['JoggingTrack'], features['LiftAvailable'],
                                features['BED'], features['VaastuCompliant'], features['Microwave'],
                                features['GolfCourse'], features['TV'], features['DiningTable'], features['Sofa'],
                                features['Wardrobe'], features['Refrigerator']]).reshape(1, -1)

    # Perform prediction
    predicted_price = model.predict(features_array)[0]
    return predicted_price

def main():
    st.title("House Price Prediction")
    st.write("Enter the features to predict the house price:")

    area = st.number_input("Area")
    bedrooms = st.number_input("No. of Bedrooms")
    resale = st.number_input("Resale")
    maintenance_staff = st.number_input("Maintenance Staff")
    gymnasium = st.number_input("Gymnasium")
    swimming_pool = st.number_input("Swimming Pool")
    landscaped_gardens = st.number_input("Landscaped Gardens")
    jogging_track = st.number_input("Jogging Track")
    lift_available = st.number_input("Lift Available")
    bed = st.number_input("BED")
    vaastu_compliant = st.number_input("Vaastu Compliant")
    microwave = st.number_input("Microwave")
    golf_course = st.number_input("Golf Course")
    tv = st.number_input("TV")
    dining_table = st.number_input("Dining Table")
    sofa = st.number_input("Sofa")
    wardrobe = st.number_input("Wardrobe")
    refrigerator = st.number_input("Refrigerator")

    features = {
        'Area': area, 'No. of Bedrooms': bedrooms, 'Resale': resale, 'MaintenanceStaff': maintenance_staff,
        'Gymnasium': gymnasium, 'SwimmingPool': swimming_pool, 'LandscapedGardens': landscaped_gardens,
        'JoggingTrack': jogging_track, 'LiftAvailable': lift_available, 'BED': bed, 'VaastuCompliant': vaastu_compliant,
        'Microwave': microwave, 'GolfCourse': golf_course, 'TV': tv, 'DiningTable': dining_table, 'Sofa': sofa,
        'Wardrobe': wardrobe, 'Refrigerator': refrigerator
    }

    if st.button("Predict"):
        predicted_price = predict_price(features)
        st.success(f"Predicted Price: {predicted_price}")


if __name__ == "__main__":
    main()
