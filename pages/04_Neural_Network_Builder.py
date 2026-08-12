import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import random
import sys, os
from sklearn.model_selection import train_test_split

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from prediction_utils import RandomConvFeatures

# Setup Page
st.set_page_config(page_title="NN Builder", page_icon="🧠", layout="wide")

st.title("Neural Network Builder")
st.markdown("Build Deep Neural Networks (DNN) or 1D Convolutional Neural Networks (1D-CNN).")

# Attempt TensorFlow Import safely
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Input
    from tensorflow.keras.callbacks import Callback
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow is not installed in your environment! Please run `pip install tensorflow` in your terminal to use this builder.")
    st.stop()

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if 'nn_layers' not in st.session_state:
    st.session_state.nn_layers = []
    
if 'trained_model_bytes' not in st.session_state:
    st.session_state.trained_model_bytes = None

if 'training_history' not in st.session_state:
    st.session_state.training_history = []

# Keras Callback for live Streamlit plotting!
class StreamlitLivePlot(Callback):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss', 0))
        val_loss = logs.get('val_loss', None)
        
        plot_data = {'Training Loss': self.losses}
        if val_loss is not None:
            self.val_losses.append(val_loss)
            plot_data['Validation Loss'] = self.val_losses
            
        df_plot = pd.DataFrame(plot_data)
        self.placeholder.line_chart(df_plot)

# -------------------------------------------------------------
# EAGER TRAINING (macOS / Apple-Silicon workaround)
# -------------------------------------------------------------
# On this TensorFlow build, Keras `model.fit()` / `train_on_batch()` DEADLOCK:
# the tf.function-compiled training step hangs at the first optimizer-variable
# init (process sits at 0% CPU forever). Pure EAGER execution — forward pass,
# GradientTape, optimizer.apply_gradients — runs fine, so we drive the training
# loop ourselves. This is mathematically equivalent to fit() and still feeds the
# live loss chart.
def _mk_loss(loss_name):
    if loss_name == "mae":
        return lambda pred, target: tf.reduce_mean(tf.abs(pred - target))
    return lambda pred, target: tf.reduce_mean(tf.square(pred - target))


def _eager_fit(model, optimizer_name, learning_rate, loss_name,
               X_train, y_train, X_val, y_val, epochs, batch_size, placeholder):
    """Train `model` with an eager GradientTape loop. Returns (train_hist, val_hist)."""
    opt = tf.keras.optimizers.get(optimizer_name)
    opt.learning_rate = learning_rate
    loss_f = _mk_loss(loss_name)

    Xt = tf.convert_to_tensor(X_train, dtype=tf.float32)
    yt = tf.convert_to_tensor(np.asarray(y_train).reshape(len(y_train), -1), dtype=tf.float32)
    has_val = X_val is not None
    if has_val:
        Xv = tf.convert_to_tensor(X_val, dtype=tf.float32)
        yv = tf.convert_to_tensor(np.asarray(y_val).reshape(len(y_val), -1), dtype=tf.float32)

    n = int(Xt.shape[0])
    bs = max(1, int(batch_size))
    train_hist, val_hist = [], []
    for _ in range(int(epochs)):
        order = np.random.permutation(n)
        for i in range(0, n, bs):
            b = order[i:i + bs]
            xb, yb = tf.gather(Xt, b), tf.gather(yt, b)
            with tf.GradientTape() as tape:
                loss = loss_f(model(xb, training=True), yb)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        # End-of-epoch metrics on the full sets (eager, no tf.function).
        train_hist.append(float(loss_f(model(Xt, training=False), yt)))
        plot = {"Training Loss": train_hist}
        if has_val:
            val_hist.append(float(loss_f(model(Xv, training=False), yv)))
            plot["Validation Loss"] = val_hist
        placeholder.line_chart(pd.DataFrame(plot))
    return train_hist, val_hist


def _eager_eval(model, loss_name, X_eval, y_eval):
    """Eager loss + MAE on an evaluation set. Returns (loss, mae)."""
    Xe = tf.convert_to_tensor(X_eval, dtype=tf.float32)
    ye = tf.convert_to_tensor(np.asarray(y_eval).reshape(len(y_eval), -1), dtype=tf.float32)
    pred = model(Xe, training=False)
    loss = float(_mk_loss(loss_name)(pred, ye))
    mae = float(tf.reduce_mean(tf.abs(pred - ye)))
    return loss, mae


# -------------------------------------------------------------
# DATA INGESTION
# -------------------------------------------------------------
st.header("1. Select Dataset")
data_source = st.radio("Choose Data Source", ["Upload CSV", "Use Session State (Preprocessed)"], horizontal=True)

df = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Spectral Data (CSV or Excel)", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.success(f"{uploaded_file.name} loaded successfully!")
        except Exception as e:
            st.error(f"Could not read file: {e}")
elif data_source == "Use Session State (Preprocessed)":
    # The Preprocessing page stores features in st.session_state.X and targets
    # in st.session_state.y (there is no single 'preprocessed_data' frame), so
    # combine them into one DataFrame for target selection below.
    X_pre = st.session_state.get("X")
    y_pre = st.session_state.get("y")
    if X_pre is not None and y_pre is not None and hasattr(X_pre, "shape"):
        try:
            X_df = (X_pre if hasattr(X_pre, "columns") else pd.DataFrame(X_pre)).copy()
            X_df.columns = [str(c) for c in X_df.columns]
            if hasattr(y_pre, "columns"):
                y_df = y_pre.copy()
                y_df.columns = [str(c) for c in y_df.columns]
            else:
                y_arr = np.asarray(y_pre)
                if y_arr.ndim == 1:
                    y_df = pd.DataFrame({"target": y_arr})
                else:
                    y_df = pd.DataFrame(y_arr, columns=[f"target_{i}" for i in range(y_arr.shape[1])])
            df = pd.concat([X_df.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)
            st.success(f"Session State Data Linked! Features: {X_df.shape[1]}, Target(s): {y_df.shape[1]}")
        except Exception as e:
            st.warning(f"Could not load session-state data: {e}. Please upload a CSV instead.")
    else:
        st.warning("No preprocessed data found in session state. Run the Preprocessing page first, or upload a CSV.")

target_cols = None
X, y = None, None
n_targets = 1
if df is not None:
    target_cols = st.multiselect(
        "Select Target Variable(s) (Y)", list(df.columns),
        help="Select one or more target columns. Multiple targets train a single "
             "multi-output network."
    )
    if target_cols:
        # Coerce to numeric float32 (features may be object dtype when they come
        # from session state or an Excel sheet), and drop rows with NaN in any
        # target OR any feature so the network never receives NaN/object data.
        X_df = df.drop(columns=target_cols).apply(pd.to_numeric, errors="coerce")
        y_df = df[target_cols].apply(pd.to_numeric, errors="coerce")
        dropped_cols = [c for c in X_df.columns if X_df[c].isna().all()]
        if dropped_cols:
            X_df = X_df.drop(columns=dropped_cols)
            st.warning(f"Dropped non-numeric feature column(s): {', '.join(map(str, dropped_cols))}")
        X = X_df.values.astype("float32")
        y = y_df.values.astype("float32")                      # (n_samples, n_targets)
        valid_idx = ~(np.isnan(y).any(axis=1) | np.isnan(X).any(axis=1))
        X = X[valid_idx]
        y = y[valid_idx]
        n_targets = y.shape[1]
        if n_targets == 1:
            y = y.ravel()                                      # keep single-target 1-D
        if X.shape[0] == 0 or X.shape[1] == 0:
            st.error("No valid numeric data after cleaning. Check your target/feature columns.")
            X, y = None, None
        else:
            st.write(f"Data Shape: **{X.shape}** samples | Target(s): **{n_targets}** {list(target_cols)}")

st.markdown("---")

tab1, tab2 = st.tabs(["Visual Block Builder (Manual)", "AutoML Tuner (Automated)"])

with tab1:
    # -------------------------------------------------------------
    # BLOCK BUILDER UI
    # -------------------------------------------------------------
    st.header("2. Architecture Builder")
    st.markdown("Add layers sequentially to build your model. An Input Layer will be automatically mapped to your data shape.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Network Stack")
        if not st.session_state.nn_layers:
            st.info("Your network is empty. Add a layer from the panel on the right.")
            
        for i, layer in enumerate(st.session_state.nn_layers):
            with st.container():
                # Card UI
                st.markdown(f"**Layer {i+1}: {layer['type']}**")
                
                c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
                with c1:
                    if layer['type'] == 'Dense':
                        st.caption(f"Units: {layer['units']} | Activation: {layer['activation']}")
                    elif layer['type'] == 'Conv1D':
                        st.caption(f"Filters: {layer['filters']} | Kernel Size: {layer['kernel_size']} | Activation: {layer['activation']}")
                    elif layer['type'] == 'MaxPooling1D':
                        st.caption(f"Pool Size: {layer['pool_size']}")
                    elif layer['type'] == 'Dropout':
                        st.caption(f"Rate: {layer['rate']}")
                    elif layer['type'] == 'Flatten':
                        st.caption("Flattens multi-dimensional arrays natively.")
                
                with c2:
                    if st.button("▲ Up", key=f"up_{i}") and i > 0:
                        st.session_state.nn_layers.insert(i-1, st.session_state.nn_layers.pop(i))
                        st.rerun()
                with c3:
                    if st.button("▼ Down", key=f"down_{i}") and i < len(st.session_state.nn_layers)-1:
                        st.session_state.nn_layers.insert(i+1, st.session_state.nn_layers.pop(i))
                        st.rerun()
                with c4:
                    if st.button("🗑️ Del", key=f"del_{i}"):
                        st.session_state.nn_layers.pop(i)
                        st.rerun()
                st.markdown("---")

    with col2:
        st.subheader("Add New Layer")
        with st.form("add_layer_form"):
            layer_type = st.selectbox("Layer Type", ["Dense", "Conv1D", "MaxPooling1D", "Dropout", "Flatten"])
            
            units, filters, kernel_size, pool_size = 0, 0, 0, 0
            rate = 0.0
            activation = "relu"
            
            if layer_type == "Dense":
                units = st.number_input("Neurons (Units)", min_value=1, value=64, step=8)
                activation = st.selectbox("Activation", ["relu", "linear", "tanh", "sigmoid"])
            elif layer_type == "Conv1D":
                filters = st.number_input("Filters", min_value=1, value=32, step=8)
                kernel_size = st.number_input("Kernel Size", min_value=1, value=3, step=1)
                activation = st.selectbox("Activation", ["relu", "tanh"], key="conv_act")
            elif layer_type == "MaxPooling1D":
                pool_size = st.number_input("Pool Size", min_value=1, value=2, step=1)
            elif layer_type == "Dropout":
                rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.9, value=0.2, step=0.05)
                
            submitted = st.form_submit_button("➕ Add Layer")
            if submitted:
                new_layer = {
                    'type': layer_type,
                    'units': units,
                    'filters': filters,
                    'kernel_size': kernel_size,
                    'pool_size': pool_size,
                    'rate': rate,
                    'activation': activation
                }
                st.session_state.nn_layers.append(new_layer)
                st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------
    # TRAINING ENGINE
    # -------------------------------------------------------------
    st.header("3. Train & Export Model")

    t_col1, t_col2 = st.columns([1, 2])

    with t_col1:
        epochs = st.number_input("Epochs", min_value=1, value=50, step=10)
        batch_size = st.number_input("Batch Size", min_value=1, value=32, step=8)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
        loss_fn = st.selectbox("Loss Function", ["mse", "mae"])
        optimizer = st.selectbox("Optimizer", ["adam", "rmsprop", "sgd"])

        m_test_size = st.slider(
            "Test set fraction (held out)", min_value=0.1, max_value=0.4, value=0.2, step=0.05,
            help="Scored only once, after training. Never used to fit or monitor the model.",
        )
        m_val_size = st.slider(
            "Validation fraction (of remaining train)", min_value=0.0, max_value=0.4, value=0.2, step=0.05,
            help="Split from the training data to monitor the live loss curve. "
                 "Set to 0 to train on all non-test data with no validation curve.",
        )

        train_clicked = st.button("Compile & Train Network", use_container_width=True, type="primary")

    with t_col2:
        loss_placeholder = st.empty()

    if train_clicked:
        if X is None or y is None:
            st.error("Please load and select a dataset and target variable first.")
        elif not st.session_state.nn_layers:
            st.error("Please add at least one layer to the network.")
        else:
            with st.spinner("Compiling and Training Neural Network..."):
                try:
                    tf.keras.backend.clear_session()
                    has_conv = any(l['type'] in ['Conv1D', 'MaxPooling1D'] for l in st.session_state.nn_layers)

                    # Three-way split: TEST is held out and scored only once at the
                    # end. VALIDATION (carved from the remaining train data) is used
                    # only to monitor the live loss curve — it never updates weights.
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=float(m_test_size), random_state=42
                    )
                    X_val = y_val = None
                    if float(m_val_size) > 0 and X_train.shape[0] > 4:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train, y_train, test_size=float(m_val_size), random_state=42
                        )

                    if has_conv:
                        X_train = np.expand_dims(X_train, axis=2)
                        X_test = np.expand_dims(X_test, axis=2)
                        if X_val is not None:
                            X_val = np.expand_dims(X_val, axis=2)
                        input_shape = (X_train.shape[1], 1)
                    else:
                        input_shape = (X_train.shape[1],)

                    model = Sequential()
                    model.add(Input(shape=input_shape))
                    
                    for layer in st.session_state.nn_layers:
                        if layer['type'] == 'Dense':
                            model.add(Dense(layer['units'], activation=layer['activation']))
                        elif layer['type'] == 'Conv1D':
                            model.add(Conv1D(layer['filters'], layer['kernel_size'], activation=layer['activation'], padding='same'))
                        elif layer['type'] == 'MaxPooling1D':
                            model.add(MaxPooling1D(layer['pool_size'], padding='same'))
                        elif layer['type'] == 'Dropout':
                            model.add(Dropout(layer['rate']))
                        elif layer['type'] == 'Flatten':
                            model.add(Flatten())
                    
                    # Output layer sized to the number of targets (multi-output).
                    n_outputs = 1 if y_train.ndim == 1 else y_train.shape[1]
                    model.add(Dense(n_outputs, activation='linear'))

                    # Train with an eager GradientTape loop (see _eager_fit).
                    # Keras fit()/train_on_batch() deadlock on this TF build, so we
                    # cannot use them. Validation (if any) is used only to draw the
                    # live loss curve; the test set is never seen during training.
                    _eager_fit(
                        model, optimizer, learning_rate, loss_fn,
                        X_train, y_train, X_val, y_val,
                        epochs, batch_size, loss_placeholder,
                    )

                    # Final, single evaluation on the untouched held-out test set.
                    test_loss, test_mae = _eager_eval(model, loss_fn, X_test, y_test)
                    st.success(f"Training Complete! Held-out Test Loss ({loss_fn.upper()}): {test_loss:.4f}")
                    mt1, mt2, mt3 = st.columns(3)
                    mt1.metric(f"Test {loss_fn.upper()}", f"{test_loss:.4f}")
                    mt2.metric("Test MAE", f"{test_mae:.4f}")
                    mt3.metric(
                        "Split (train / val / test)",
                        f"{X_train.shape[0]} / {0 if X_val is None else X_val.shape[0]} / {X_test.shape[0]}",
                    )
                    
                    # Extract parameters for PKL compatibility
                    model_json = model.to_json()
                    model_weights = model.get_weights()
                    
                    payload = {
                        'architecture_json': model_json,
                        'weights': model_weights,
                        'model_type': 'keras_custom',
                        'input_features_count': X_train.shape[1]
                    }
                    
                    buffer = io.BytesIO()
                    pickle.dump(payload, buffer)
                    st.session_state.trained_model_bytes = buffer.getvalue()
                    
                except Exception as e:
                    st.error(f"Architecture Error: {str(e)}")
                    st.warning("Hint: Did you forget to add a 'Flatten' layer after Conv1D before placing Dense layers?")

with tab2:
    st.header("AutoML Tuner (Random Search + Cross-Validation)")
    st.markdown(
        "Automatically search for the best neural network. Hyperparameters are "
        "selected by **k-fold cross-validation on the training set only**; the "
        "**test set is held out and scored exactly once** at the end — so the "
        "reported test metrics are an honest, leakage-free estimate of "
        "generalization. Fast and robust (no TensorFlow/GPU), and the result is a "
        "standard scikit-learn model that works directly on the Prediction page."
    )

    a_col1, a_col2 = st.columns([1, 2])
    with a_col1:
        arch_choice = st.radio(
            "Architecture",
            ["MLP (Dense)", "1D-CNN (Conv features + MLP)"],
            help="MLP: a fully-connected network. 1D-CNN: random 1D convolution "
                 "kernels extract local, shift-invariant patterns (ROCKET-style) "
                 "before the MLP — CNN-like, but with no TensorFlow.",
        )
        num_trials = st.number_input("Number of Trials", min_value=2, max_value=100, value=15, step=1)
        cv_folds = st.number_input(
            "Cross-validation folds", min_value=2, max_value=10, value=5, step=1,
            help="Hyperparameters are selected by k-fold CV on the TRAINING set "
                 "only. The test set is held out and scored just once at the end.",
        )
        test_size = st.slider(
            "Test set fraction (held out)", min_value=0.1, max_value=0.4, value=0.2, step=0.05,
        )
        max_iter = st.number_input("Max iterations per trial", min_value=50, max_value=2000, value=300, step=50)
        n_kernels = 200
        if arch_choice.startswith("1D-CNN"):
            n_kernels = st.number_input("Conv kernels", min_value=32, max_value=1000, value=200, step=32)
        auto_train = st.button("Start AutoML Tuning", use_container_width=True, type="primary")

    with a_col2:
        auto_status = st.empty()
        auto_progress = st.progress(0)
        best_metric_display = st.empty()

    if auto_train:
        if X is None or y is None:
            st.error("Please load and select a dataset and target variable first.")
        else:
            try:
                from sklearn.neural_network import MLPRegressor
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                from sklearn.model_selection import cross_val_score, KFold

                use_conv = arch_choice.startswith("1D-CNN")

                # ---- Hold out the test set. It is NOT touched during tuning; it
                # is scored exactly once, at the very end, for an honest estimate.
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=float(test_size), random_state=42
                )

                # Guard: need enough training samples for the requested folds.
                k = int(cv_folds)
                if X_train.shape[0] < k * 2:
                    k = max(2, X_train.shape[0] // 2)
                    auto_status.warning(f"Too few training samples for {int(cv_folds)} folds; using {k}.")

                # For the 1D-CNN option, extract random-conv features ONCE from the
                # TRAINING data (kernels are random/unsupervised, so this is leakage
                # free) and search over the MLP on those features — keeps CV fast.
                if use_conv:
                    auto_status.info("Extracting 1D convolution features (train only)...")
                    conv_extractor = RandomConvFeatures(n_kernels=int(n_kernels), random_state=42)
                    X_train_feat = conv_extractor.fit_transform(X_train)
                else:
                    X_train_feat = X_train

                best_score = -np.inf          # best mean CV R² on TRAIN
                best_mlp_kwargs = None
                best_config = {}

                hidden_choices = [(64,), (128,), (256,), (128, 64), (256, 128), (128, 64, 32)]
                alpha_choices = [1e-4, 1e-3, 1e-2, 1e-1]
                lr_choices = [1e-2, 5e-3, 1e-3, 5e-4]
                act_choices = ["relu", "tanh"]

                cv = KFold(n_splits=k, shuffle=True, random_state=42)
                n_trials = int(num_trials)
                for trial in range(n_trials):
                    auto_status.info(f"Running Trial {trial+1} / {n_trials} ({k}-fold CV on train)...")
                    auto_progress.progress(trial / n_trials)

                    mlp_kwargs = dict(
                        hidden_layer_sizes=random.choice(hidden_choices),
                        activation=random.choice(act_choices),
                        alpha=random.choice(alpha_choices),
                        learning_rate_init=random.choice(lr_choices),
                        max_iter=int(max_iter),
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=42,
                    )
                    model = Pipeline([("scaler", StandardScaler()),
                                      ("mlp", MLPRegressor(**mlp_kwargs))])
                    try:
                        # Model selection uses ONLY the training set, via CV.
                        cv_scores = cross_val_score(
                            model, X_train_feat, y_train, cv=cv, scoring="r2"
                        )
                        score = float(np.mean(cv_scores))
                    except Exception:
                        continue  # skip a trial that fails to converge

                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_mlp_kwargs = mlp_kwargs
                        best_config = {"architecture": arch_choice,
                                       "hidden_layers": mlp_kwargs["hidden_layer_sizes"],
                                       "alpha": mlp_kwargs["alpha"],
                                       "learning_rate": mlp_kwargs["learning_rate_init"],
                                       "activation": mlp_kwargs["activation"]}
                        best_metric_display.success(
                            f"🔥 New best CV R² (train): **{best_score:.4f}**  ({best_config})"
                        )

                auto_progress.progress(1.0)
                if best_mlp_kwargs is not None:
                    # Rebuild the full, exportable pipeline on RAW X so the saved
                    # model includes the conv feature step (works on new raw data).
                    steps = []
                    if use_conv:
                        steps.append(("conv", RandomConvFeatures(n_kernels=int(n_kernels), random_state=42)))
                    steps.append(("scaler", StandardScaler()))
                    steps.append(("mlp", MLPRegressor(**best_mlp_kwargs)))
                    final_model = Pipeline(steps)
                    # Fit the winning configuration on the FULL training set...
                    final_model.fit(X_train, y_train)

                    # ...and evaluate ONCE on the untouched held-out test set.
                    y_pred = final_model.predict(X_test)
                    test_r2 = r2_score(y_test, y_pred)
                    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    test_mae = float(mean_absolute_error(y_test, y_pred))

                    buffer = io.BytesIO()
                    pickle.dump(final_model, buffer)
                    st.session_state.trained_model_bytes = buffer.getvalue()

                    auto_status.success("AutoML Tuning Complete!")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Best CV R² (train)", f"{best_score:.4f}")
                    m2.metric("Held-out Test R²", f"{test_r2:.4f}")
                    m3.metric("Test RMSE", f"{test_rmse:.4f}")
                    m4.metric("Test MAE", f"{test_mae:.4f}")
                    st.caption(
                        f"Best config: {best_config} — selected by {k}-fold CV on "
                        f"{X_train.shape[0]} training samples, then evaluated once on "
                        f"{X_test.shape[0]} held-out test samples."
                    )
                else:
                    auto_status.error("No valid model found. Check that your target is numeric "
                                      "and that you have enough samples.")
            except Exception as e:
                import traceback
                auto_status.error(f"AutoML tuning failed: {e}")
                st.code(traceback.format_exc())


# -------------------------------------------------------------
# PKL EXPORT (Shared for both tabs)
# -------------------------------------------------------------
st.markdown("---")
if st.session_state.trained_model_bytes is not None:
    st.markdown("### Export Your Trained Network")
    st.download_button(
        label="Download Best Trained Model (.pkl)",
        data=st.session_state.trained_model_bytes,
        file_name="trained_deep_network.pkl",
        mime="application/octet-stream",
        type="primary"
    )
    st.info(
        "The .pkl holds the best model from the Manual Builder (a Keras architecture "
        "+ weights payload) or the AutoML Tuner (a ready-to-use scikit-learn MLP "
        "pipeline that the Prediction page can load directly)."
    )
