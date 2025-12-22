"""
Streamlit app for generating personalized storybooks
"""
import streamlit as st
import os
from pathlib import Path
from PIL import Image
import io
import tempfile
from config import TEMPLATE_IMAGES_DIR, TOTAL_PAGES, BOOKS_BASE_DIR, OPENAI_API_KEY, get_openai_api_key
from openai_client_new import generate_pages_for_book, process_cover_with_new_workflow, generate_canonical_reference
from generate_pdf import create_pdf
from book_utils import BOOK_PATHS, get_book_template_path

# Debug: Check API key on app load (dynamically check all sources)
try:
    current_api_key = get_openai_api_key()
    if current_api_key:
        # Determine key source (priority order: session state > secrets > env > hardcoded)
        key_source = "Hardcoded Fallback"
        streamlit_key_truncated = False
        
        # Check session state first
        session_key = st.session_state.get("openai_api_key", None)
        if session_key and str(session_key).strip() == current_api_key:
            key_source = "User Input (Session State)"
        else:
            # Check Streamlit secrets
            try:
                if hasattr(st, 'secrets'):
                    try:
                        st_key = st.secrets.get("OPENAI_API_KEY", None)
                        if st_key:
                            st_key_clean = str(st_key).strip().replace('\n', '').replace('\r', '')
                            if st_key_clean == current_api_key:
                                key_source = "Streamlit Secret"
                            elif len(st_key_clean) < 200:
                                streamlit_key_truncated = True
                    except Exception:
                        pass
            except Exception:
                pass
            
            # Check environment variable
            if key_source == "Hardcoded Fallback":
                import os
                env_key = os.getenv("OPENAI_API_KEY")
                if env_key and env_key == current_api_key:
                    key_source = "Environment Variable"
        
        # Show status (don't show error for short keys if from user input - they may be valid)
        if key_source == "User Input (Session State)":
            if len(current_api_key) >= 50 and current_api_key.startswith("sk-"):
                st.sidebar.success(f"✅ API Key loaded from user input ({len(current_api_key)} chars)")
            else:
                st.sidebar.warning(f"⚠️ API Key format may be invalid ({len(current_api_key)} chars)")
        elif len(current_api_key) < 200 and key_source != "User Input (Session State)":
            st.sidebar.error(f"❌ API Key is TRUNCATED! ({len(current_api_key)} chars)")
            st.sidebar.warning("📝 Enter your API key in the input field above, or configure it in Streamlit Cloud secrets")
            st.sidebar.info("🔗 Get your key: https://platform.openai.com/account/api-keys")
        else:
            if streamlit_key_truncated:
                st.sidebar.warning(f"⚠️ Streamlit secret was truncated, using fallback key ({len(current_api_key)} chars)")
            else:
                st.sidebar.success(f"✅ API Key loaded ({key_source}, {len(current_api_key)} chars)")
    else:
        st.sidebar.error("❌ API Key not found! Please enter it in the input field above.")
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not verify API key: {e}")

# Page configuration
st.set_page_config(
    page_title="Personalized Storybook Generator",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = None
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = "A True Princess"
# Canonical reference for identity consistency across all pages
if 'canonical_reference' not in st.session_state:
    st.session_state.canonical_reference = None
if 'identity_info' not in st.session_state:
    st.session_state.identity_info = None

# Sidebar for inputs
with st.sidebar:
    st.header("📝 Input")
    
    # API Key input (highest priority - user can enter their own key)
    api_key_input = st.text_input(
        "🔑 OpenAI API Key",
        value=st.session_state.get("openai_api_key", ""),
        type="password",
        help="Enter your OpenAI API key. This will be used for all API calls. Get your key from https://platform.openai.com/account/api-keys",
        placeholder="sk-proj-..."
    )
    
    # Store in session state if provided
    if api_key_input:
        api_key_clean = api_key_input.strip()
        # Validate format
        if api_key_clean and len(api_key_clean) >= 50 and api_key_clean.startswith("sk-"):
            st.session_state.openai_api_key = api_key_clean
            st.success("✅ API Key format valid")
        elif api_key_clean:
            st.session_state.openai_api_key = api_key_clean  # Store anyway, let validation happen later
            if not api_key_clean.startswith("sk-"):
                st.warning("⚠️ API key should start with 'sk-'")
            elif len(api_key_clean) < 50:
                st.warning(f"⚠️ API key seems too short ({len(api_key_clean)} chars)")
    else:
        # Clear session state if input is empty
        if "openai_api_key" in st.session_state:
            del st.session_state.openai_api_key
    
    st.markdown("---")
    
    # Book selection - must be first in sidebar
    book_options = list(BOOK_PATHS.keys())
    current_index = book_options.index(st.session_state.selected_book) if st.session_state.selected_book in book_options else 0
    
    selected_book = st.selectbox(
        "📚 Choose Storybook",
        options=book_options,
        index=current_index,
        help="Select which storybook to personalize"
    )
    st.session_state.selected_book = selected_book
    
    # Update title based on selected book
    book_emoji = {
        "A True Princess": "👑",
        "My Animal World": "🦁",
        "My Dinosaur World": "🦕",
        "My Little Wonder": "✨",
        "Superhero": "🦸",
        "Little Farmer's Big Day": "🚜"
    }
    st.markdown(f"**Selected:** {book_emoji.get(selected_book, '📚')} {selected_book}")
    
    # File uploader for child's photo
    uploaded_file = st.file_uploader(
        "Upload child's photo",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear photo of the child's face"
    )
    
    # Display uploaded image (read bytes first, then display)
    if uploaded_file is not None:
        # Read the file bytes first
        uploaded_file.seek(0)  # Reset to beginning
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset again for later use
        
        # Display the image
        child_image = Image.open(io.BytesIO(file_bytes))
        st.image(child_image, caption="Child's Photo", width='stretch')
    
    # Child's name input
    child_name = st.text_input(
        "Child's Name",
        placeholder="Enter the child's name",
        help="This will be used for the PDF filename"
    )

    # Generation scope (cover only, specific pages, or full book)
    generation_mode = st.radio(
        "Generation mode",
        ["Cover only", "Specific pages", "Full book"],
        index=0,
        help="Choose whether to personalize just the cover, specific pages, or all pages"
    )
    
    # Text input for specific pages (only shown when "Specific pages" is selected)
    specific_pages_input = ""
    if generation_mode == "Specific pages":
        specific_pages_input = st.text_input(
            "Enter page numbers",
            placeholder="e.g., 1, 5, 6, 10",
            help="Enter comma-separated page numbers. Include '1' for cover, or use 2-33 for interior pages only"
        )
    
    # Calculate pages to generate based on mode
    if generation_mode == "Full book":
        pages_to_generate = TOTAL_PAGES
        mode_description = "33-page storybook"
        time_estimate = "5-11 minutes"
    elif generation_mode == "Specific pages":
        # Parse specific pages
        if specific_pages_input:
            try:
                specific_pages = [int(p.strip()) for p in specific_pages_input.split(",") if p.strip()]
                # Check if cover (page 1) is included
                include_cover = 1 in specific_pages
                interior_pages = [p for p in specific_pages if 2 <= p <= TOTAL_PAGES]
                pages_to_generate = (1 if include_cover else 0) + len(interior_pages)
                
                if include_cover and interior_pages:
                    mode_description = f"cover + pages {', '.join(map(str, interior_pages))}"
                elif include_cover:
                    mode_description = "cover only"
                else:
                    mode_description = f"pages {', '.join(map(str, interior_pages))}"
                
                time_estimate = f"{pages_to_generate}-{pages_to_generate * 2} minutes"
            except ValueError:
                specific_pages = []
                pages_to_generate = 0
                mode_description = "no pages (invalid page numbers)"
                time_estimate = "0 minutes"
        else:
            pages_to_generate = 0
            mode_description = "no pages (no pages specified)"
            time_estimate = "0 minutes"
    else:  # Cover only
        pages_to_generate = 1
        mode_description = "cover only"
        time_estimate = "1-3 minutes"
    

# Main content area
col1, col2 = st.columns([2, 1])

# Update main title based on selected book
book_emoji = {
    "A True Princess": "👑",
    "My Animal World": "🦁",
    "My Dinosaur World": "🦕",
    "My Little Wonder": "✨",
    "Superhero": "🦸",
    "Little Farmer's Big Day": "🚜"
}
selected_book = st.session_state.selected_book
st.title(f"{book_emoji.get(selected_book, '📚')} Personalized Storybook Generator")
st.markdown(f"Upload a child's photo to create a personalized **{selected_book}** storybook!")

with col1:
    if uploaded_file and child_name:
        if st.button("✨ Generate Storybook", type="primary", width='stretch'):
            with st.spinner("Preparing..."):
                # Read child photo (reset file pointer first)
                uploaded_file.seek(0)
                child_photo_bytes = uploaded_file.read()
                
                # Verify we got the image
                if len(child_photo_bytes) < 1000:  # Less than 1KB is suspicious
                    st.error(f"Error: Child photo appears to be empty or corrupted ({len(child_photo_bytes)} bytes). Please upload the image again.")
                    st.stop()
                
                # Get template directory for selected book
                template_dir = get_book_template_path(selected_book)
                if not template_dir.exists():
                    st.error(f"Template directory not found for {selected_book}: {template_dir}")
                    st.stop()
                
                # Check for new cover workflow files (00.png, 0.png, 1.png)
                front_cover_path = template_dir / "00.png"  # Front cover only
                back_cover_path = template_dir / "0.png"    # Back cover only
                full_cover_path = template_dir / "1.png"    # Full wrap (for spine extraction)
                
                use_new_cover_workflow = (
                    front_cover_path.exists() and 
                    back_cover_path.exists() and 
                    full_cover_path.exists()
                )
                
                # Prepare interior page list based on generation mode
                # Also determine if cover should be generated
                interior_page_list = []
                should_generate_cover = False
                
                if generation_mode == "Full book":
                    # Generate all interior pages (2-33) and cover
                    should_generate_cover = True
                    for i in range(2, TOTAL_PAGES + 1):  # Start from page 2 (interior pages)
                        page_path = template_dir / f"{i}.png"
                        if page_path.exists():
                            interior_page_list.append(f"{i}.png")
                        else:
                            st.warning(f"Page {i}.png not found, skipping...")
                elif generation_mode == "Specific pages" and specific_pages_input:
                    # Generate only the specified pages
                    try:
                        specific_pages = [int(p.strip()) for p in specific_pages_input.split(",") if p.strip()]
                        # Check if cover (page 1) is requested
                        if 1 in specific_pages:
                            should_generate_cover = True
                            specific_pages.remove(1)  # Remove 1 from list since it's handled separately
                        
                        for i in specific_pages:
                            if 2 <= i <= TOTAL_PAGES:  # Valid interior page range
                                page_path = template_dir / f"{i}.png"
                                if page_path.exists():
                                    interior_page_list.append(f"{i}.png")
                                else:
                                    st.warning(f"Page {i}.png not found, skipping...")
                            else:
                                st.warning(f"Page {i} is out of range (valid: 2-{TOTAL_PAGES}), skipping...")
                    except ValueError:
                        st.error("Invalid page numbers. Please enter comma-separated numbers like: 5, 6, 10")
                        st.stop()
                elif generation_mode == "Cover only":
                    # Only generate cover
                    should_generate_cover = True
                
                # Validation
                if not interior_page_list and not should_generate_cover:
                    st.error("No pages selected to generate!")
                    st.stop()

                if should_generate_cover and not use_new_cover_workflow:
                    st.error("Cover generation requires the new cover workflow files (00.png, 0.png, 1.png).")
                    st.stop()
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate total steps (cover + optional interior pages)
            total_steps = (1 if should_generate_cover and use_new_cover_workflow else 0) + len(interior_page_list)
            progress_state = {"current_step": 0}
            
            def progress_callback(step, msg):
                progress_state["current_step"] = step
                status_text.text(msg)
                if total_steps > 0:
                    progress_bar.progress(min(1.0, progress_state["current_step"] / total_steps))
            
            try:
                generated_images = []
                original_dimensions = []
                
                # =========================================================
                # STEP 0: GENERATE CANONICAL REFERENCE (identity anchor)
                # This ensures consistent identity across ALL pages
                # =========================================================
                status_text.text("Step 1/3: Generating canonical reference portrait...")
                
                # Create placeholder for canonical reference preview
                canonical_preview_placeholder = st.empty()
                
                try:
                    # Verify API key before making calls (use dynamic function)
                    current_api_key = get_openai_api_key()
                    if not current_api_key or current_api_key == "":
                        st.error("❌ OpenAI API Key is not configured! Please add it to Streamlit Cloud secrets.")
                        st.stop()
                    
                    status_text.text("Step 1/3: Generating canonical reference portrait... (This may take 30-60 seconds)")
                    st.info(f"🔑 API Key status: ✅ Loaded (using dynamic retrieval)")
                    
                    canonical_reference_bytes, identity_info = generate_canonical_reference(
                        child_image_bytes=child_photo_bytes,
                        book_name=selected_book
                    )
                    
                    # Store in session state
                    st.session_state.canonical_reference = canonical_reference_bytes
                    st.session_state.identity_info = identity_info
                    
                    # Show canonical reference preview
                    with canonical_preview_placeholder.container():
                        st.subheader("🎯 Canonical Reference (Identity Anchor)")
                        col_ref1, col_ref2 = st.columns(2)
                        with col_ref1:
                            st.image(Image.open(io.BytesIO(child_photo_bytes)), caption="Original Photo", width='stretch')
                        with col_ref2:
                            st.image(Image.open(io.BytesIO(canonical_reference_bytes)), caption="Canonical Reference", width='stretch')
                        st.caption("This reference portrait will be used for consistent identity across all pages.")
                    
                    status_text.text("Canonical reference generated! Processing pages...")
                    
                except Exception as ref_error:
                    error_msg = str(ref_error)
                    st.error(f"❌ Error generating canonical reference: {error_msg}")
                    st.exception(ref_error)  # Show full traceback
                    st.warning("Using original photo as fallback. API calls may not be working.")
                    canonical_reference_bytes = child_photo_bytes
                    identity_info = None
                
                # Step 1: Process cover using new workflow (if requested and available)
                if should_generate_cover and use_new_cover_workflow:
                    try:
                        status_text.text("Step 2/3: Processing cover (front cover personalization)... (This may take 1-2 minutes)")
                        
                        # Read cover files
                        front_cover_bytes = front_cover_path.read_bytes()
                        back_cover_bytes = back_cover_path.read_bytes()
                        full_cover_bytes = full_cover_path.read_bytes()
                        
                        # Process cover with new workflow
                        cover_bytes, cover_dims = process_cover_with_new_workflow(
                            child_photo_bytes,
                            front_cover_bytes,
                            back_cover_bytes,
                            full_cover_bytes,
                            child_name=child_name,
                            book_name=selected_book
                        )
                        
                        generated_images.append(cover_bytes)
                        original_dimensions.append(cover_dims)
                        current_step = 1
                        next_msg = "Cover processed! Processing interior pages..." if interior_page_list else "Cover processed!"
                        progress_callback(current_step, next_msg)
                    except Exception as cover_error:
                        st.error(f"❌ Error processing cover: {str(cover_error)}")
                        st.exception(cover_error)
                        raise
                
                # Step 2: Process interior pages (pages 2-33) with canonical reference
                if interior_page_list:
                    status_text.text("Step 3/3: Processing interior pages with canonical reference...")
                    
                    def interior_progress_callback(step, msg):
                        # Offset by 1 for cover step if cover was generated
                        offset = 1 if (should_generate_cover and use_new_cover_workflow) else 0
                        progress_callback(offset + step, msg)
                    
                    interior_images, interior_dimensions = generate_pages_for_book(
                        child_photo_bytes,
                        str(template_dir),
                        interior_page_list,
                        status_callback=interior_progress_callback,
                        child_name=child_name,
                        canonical_reference_bytes=canonical_reference_bytes,
                        identity_info=identity_info
                    )
                    
                    generated_images.extend(interior_images)
                    original_dimensions.extend(interior_dimensions)
                
                # Update progress
                progress_bar.progress(1.0)
                status_text.text("All requested pages generated! Creating PDF...")
                
                # Create PDF with original dimensions preserved
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    pdf_path = tmp_file.name
                
                # Get base directory for PDF dimension lookup
                base_dir = BOOKS_BASE_DIR
                
                create_pdf(
                    generated_images, 
                    pdf_path, 
                    child_name, 
                    original_dimensions=original_dimensions,
                    book_name=selected_book,
                    base_dir=base_dir
                )
                
                # Store in session state
                st.session_state.generated_images = generated_images
                st.session_state.pdf_path = pdf_path
                
                status_text.text("✅ Storybook generated successfully!")
                st.success("🎉 Your personalized storybook is ready!")
                
            except Exception as e:
                st.error(f"Error generating storybook: {str(e)}")
                st.exception(e)
    
    # Display preview and download
    if st.session_state.generated_images:
        st.header("📖 Preview")
        
        # Show first few pages as preview
        preview_cols = st.columns(3)
        for i, img_bytes in enumerate(st.session_state.generated_images[:6]):
            with preview_cols[i % 3]:
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, caption=f"Page {i+1}", width='stretch')
        
        if len(st.session_state.generated_images) > 6:
            st.caption(f"... and {len(st.session_state.generated_images) - 6} more pages")
        
        # Download button
        st.markdown("---")
        if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
            with open(st.session_state.pdf_path, 'rb') as pdf_file:
                pdf_bytes = pdf_file.read()
            
            # Get selected book name for filename (sanitize for filesystem)
            book_name_safe = st.session_state.selected_book.replace(" ", "_")
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"{child_name}_{book_name_safe}.pdf",
                mime="application/pdf",
                type="primary",
                width='stretch'
            )

with col2:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. **Upload Photo**: Choose a clear photo of the child
    2. **Enter Name**: Type the child's name
    3. **Generate**: Click the generate button
    4. **Wait**: The process takes 5-11 minutes
    5. **Download**: Get your personalized PDF!
    
    ---
    
    **Note**: The app will:
    - Insert the child into all 33 pages
    - Maintain the original artistic style
    - Preserve all text and layout
    - Create a beautiful PDF storybook
    """)
