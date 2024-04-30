
import streamlit as st

css = """
<style>


h1 {
  color: #FFD700; 
  text-align: center; 
  text-shadow: 2px 2px 2px rgba(0,0,0,0.5);
}

#gif-container {
  display: flex;
  justify-content: center; 
  
  
}


</style>
"""

st.set_page_config(page_title="CinéMachine", layout="wide")




# titres
st.title("🎬 CinéMachine 🎥")
st.markdown("![Alt Text](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGNpZ2IwYjdudnV4ZGtueDBuaDdvNnRkZG5kdjNyN3FzeTczbWNsMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3orieYmP5YQmpsTB4c/giphy.gif)")
sidebar_gif_url = "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXluazhiYzd1NmdhYnU1emprd21qOGVwYTZuMXNsY21hMmE5eG5vbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/XA4s9iaViGkgJAvM3D/giphy.gif"
st.sidebar.image(sidebar_gif_url, use_column_width=True)