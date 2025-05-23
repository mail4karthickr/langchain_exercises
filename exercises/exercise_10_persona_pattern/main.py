import streamlit as st

from llm.openai import LLM

def run():
    PersonaPattern().run()

class PersonaPattern:
    def __init__(self):
        self.model = LLM().chat_groq

    def run(self):
        st.markdown(self.persona())
        if st.button("Act as a marketing manager"):
            response = self.model.invoke(self.marketing_manager_prompt())
            st.write(response.content)

    def persona(self):
        return """
        ## 🧠 Persona Pattern

        ### 📖 Definition
        The **Persona Pattern** is a prompt engineering technique where the model is instructed to assume a specific role or identity (such as a teacher, doctor, or engineer) to generate responses that match the tone, style, and expertise expected of that persona.

        ### 💡 Example
        **Prompt:**
        > You are a senior software engineer with 10 years of experience in iOS development. Explain the difference between Swift's `class` and `struct` to a junior developer.

        **Response:**
        > Sure! In Swift, both `class` and `struct` are used to create custom data types, but there are key differences:
        >
        > - `Struct` is a **value type**, meaning it's copied when passed around.
        > - `Class` is a **reference type**, meaning changes to one reference affect all references.
        > - `Structs` don't support inheritance; `classes` do.
        >
        > For most simple data models, prefer `struct`. Use `class` when you need inheritance or reference semantics.
        """
    
    def marketing_manager_prompt(self):
        return """
            Act as a marketing manager,
            your task is to help a marketing team create an eye-catching advert
            for a new smartphone product based on the following technical fact sheet.

            Create an advert where you highlight the most important features
            and briefly mention how they could be used with an example.
            Try to captivate the audience with these examples.
            Focus especially on the screen size, hardware, camera, battery life

            PRODUCT NAME: Apple iPhone 15

            PRODUCT SPECS:

            NETWORK	Technology
            GSM / CDMA / HSPA / EVDO / LTE / 5G
            LAUNCH	Announced	2023, September 12
            Status	Available. Released 2023, September 22
            BODY	Dimensions	147.6 x 71.6 x 7.8 mm (5.81 x 2.82 x 0.31 in)
            Weight	171 g (6.03 oz)
            Build	Glass front (Corning-made glass), glass back (Corning-made glass), aluminum frame
            SIM	Nano-SIM and eSIM - International
            Dual eSIM with multiple numbers - USA
            Dual SIM (Nano-SIM, dual stand-by) - China
                IP68 dust/water resistant (up to 6m for 30 min)
            Apple Pay (Visa, MasterCard, AMEX certified)
            DISPLAY	Type	Super Retina XDR OLED, HDR10, Dolby Vision, 1000 nits (HBM), 2000 nits (peak)
            Size	6.1 inches, 91.3 cm2 (~86.4% screen-to-body ratio)
            Resolution	1179 x 2556 pixels, 19.5:9 ratio (~461 ppi density)
            Protection	Ceramic Shield glass
            PLATFORM	OS	iOS 17, upgradable to iOS 17.1
            Chipset	Apple A16 Bionic (4 nm)
            CPU	Hexa-core (2x3.46 GHz Everest + 4x2.02 GHz Sawtooth)
            GPU	Apple GPU (5-core graphics)
            MEMORY	Card slot	No
            Internal	128GB 6GB RAM, 256GB 6GB RAM, 512GB 6GB RAM
                NVMe
            MAIN CAMERA	Dual	48 MP, f/1.6, 26mm (wide), 1/1.56", 1.0µm, dual pixel PDAF, sensor-shift OIS
            12 MP, f/2.4, 13mm, 120˚ (ultrawide)
            Features	Dual-LED dual-tone flash, HDR (photo/panorama)
            Video	4K@24/25/30/60fps, 1080p@25/30/60/120/240fps, HDR, Dolby Vision HDR (up to 60fps), Cinematic mode (4K@30fps), stereo sound rec.
            SELFIE CAMERA	Single	12 MP, f/1.9, 23mm (wide), 1/3.6", PDAF
            SL 3D, (depth/biometrics sensor)
            Features	HDR, Cinematic mode (4K@30fps)
            Video	4K@24/25/30/60fps, 1080p@25/30/60/120fps, gyro-EIS
            SOUND	Loudspeaker	Yes, with stereo speakers
            3.5mm jack	No
            COMMS	WLAN	Wi-Fi 802.11 a/b/g/n/ac/6, dual-band, hotspot
            Bluetooth	5.3, A2DP, LE
            Positioning	GPS, GLONASS, GALILEO, BDS, QZSS
            NFC	Yes
            Radio	No
            USB	USB Type-C 2.0, DisplayPort
            FEATURES	Sensors	Face ID, accelerometer, gyro, proximity, compass, barometer
                Ultra Wideband 2 (UWB) support
            Emergency SOS via satellite (SMS sending/receiving)
            BATTERY	Type	Li-Ion 3349 mAh, non-removable
            Charging	Wired, PD2.0, 50% in 30 min (advertised)
            15W wireless (MagSafe)
            7.5W wireless (Qi)
            4.5W reverse wired
            MISC	Colors	Black, Blue, Green, Yellow, Pink
            Models	A3090, A2846, A3089, A3092, iPhone15,4
            SAR	1.12 W/kg (head)     1.14 W/kg (body)
            SAR EU	0.98 W/kg (head)     0.98 W/kg (body)
            Price	₹ 79,900 / £ 799.00 / € 949.00
            TESTS	Performance	AnTuTu: 1354651 (v10)
            GeekBench: 6526 (v6)
            GFXBench: 60fps (ES 3.1 onscreen)
            Display	Contrast ratio: Infinite (nominal)
            Camera	Photo / Video
            Loudspeaker	-26.2 LUFS (Good)
            Battery life
            Endurance rating 98h
        """