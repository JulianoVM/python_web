# Aqui vão as rotas e os links
from tumbrl import app
from flask import render_template, url_for, redirect, session, flash
from flask_login import login_required, login_user, current_user
from tumbrl.models import load_user
from tumbrl.forms import FormLogin, FormCreateNewAccount, FormCreateNewPost
from tumbrl import bcrypt
from tumbrl.models import User, Posts
from tumbrl import database

import os
from werkzeug.utils import secure_filename


# @app.route('/home')
@app.route('/', methods=['POST', 'GET'])
def homepage():
    _formLogin = FormLogin()
    if _formLogin.validate_on_submit():
        userToLogin = User.query.filter_by(email=_formLogin.email.data).first()
        if userToLogin and bcrypt.check_password_hash(userToLogin.password, _formLogin.password.data):
            login_user(userToLogin)
            return redirect(url_for("profile", user_id=userToLogin.id))

    return render_template('home.html', form=_formLogin,)


@app.route('/new', methods=['POST', 'GET'])
def createAccount():
    _formCreateNewAccount = FormCreateNewAccount()

    if _formCreateNewAccount.validate_on_submit():
        password = _formCreateNewAccount.password.data
        password_cr = bcrypt.generate_password_hash(password)
        # print(password)
        # print(password_cr)

        newUser = User(
            username=_formCreateNewAccount.usarname.data,
            email=_formCreateNewAccount.email.data,
            password=password_cr
        )

        database.session.add(newUser)
        database.session.commit()

        # Desafio
        # Fazer Login e Mandar para a pagina de perfil dele

        login_user(newUser, remember=True)
        return redirect(url_for('profile', user_id=newUser.id))

    return render_template('new.html', form=_formCreateNewAccount)


@app.route('/perry')
def perry():
    return render_template('perry.html')


@app.route('/teste')
def teste():
    return render_template('teste.html')


@app.route('/like', methods=['POST'])
def like():
    contador = session.get('contador', 0)

    if 'clicou' not in session:
        contador += 1
        session['clicou'] = True
    else:
        contador -= 1
        session.pop('clicou', None)

    session['contador'] = contador

    user_id = current_user.id

    return redirect(url_for('profile', user_id=user_id))


@app.route('/profile/<user_id>', methods=['POST', 'GET'])
@login_required
def profile(user_id):
    contador = session.get('contador', 0)

    if int(user_id) == int(current_user.id):
        _formCreateNewPost = FormCreateNewPost()

        if _formCreateNewPost.validate_on_submit():
            photo_file = _formCreateNewPost.photo.data
            photo_name = secure_filename(photo_file.filename)

            photo_path = f'{os.path.abspath(os.path.dirname(__file__))}/{app.config["UPLOAD_FOLDER"]}/{photo_name}'
            photo_file.save(photo_path)

            _postText = _formCreateNewPost.text.data

            newPost = Posts(post_text=_postText, post_img=photo_name, user_id=int(current_user.id))
            database.session.add(newPost)
            database.session.commit()

        return render_template('profile.html', user=current_user, form=_formCreateNewPost, contador=contador)

    else:
        _user = User.query.get(int(user_id))
        return render_template('profile.html', user=_user, form=None)
    

@app.route('/delete_post/<post_id>', methods=['POST'])
@login_required
def delete_post(post_id):
    post_to_delete = Posts.query.get(post_id)

    if post_to_delete:
        if post_to_delete.user_id == current_user.id:
            database.session.delete(post_to_delete)
            database.session.commit()
            flash('Post removido com sucesso.', 'success')
        else:
            flash('Você não tem permissão para remover este post.', 'danger')
    else:
        flash('Post não encontrado.', 'danger')

    return redirect(url_for('profile', user_id=current_user.id))